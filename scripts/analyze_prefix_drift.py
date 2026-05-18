#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from textboost.text_encoders.clip import TextModel
from textboost.ti_utils import load_new_token


DEFAULT_PROMPTS = [
    "a studio photo of <INSTANCE> SUBJECT",
    "a <INSTANCE> SUBJECT in the snow",
    "a <INSTANCE> SUBJECT on a cobblestone street",
    "painting of a <INSTANCE> SUBJECT in the Monet style",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare prefix-token drift for CPA vs naive text-encoder fine-tuning."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="models/stable-diffusion-2-1-base",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="datasets/dreambooth_n1.json",
    )
    parser.add_argument("--cpa_path", type=str, required=True)
    parser.add_argument("--naive_path", type=str, required=True)
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--prompts", type=str, nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--output_csv", type=str, default="outputs/rebuttal_prefix_drift.csv"
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _placeholder_tokens(placeholder: str, num_vectors: int) -> list[str]:
    if num_vectors <= 1:
        return [placeholder]
    if placeholder.endswith(">"):
        tokens = [placeholder[:-1] + "_0>"]
        for i in range(1, num_vectors):
            tokens.append(f"{placeholder[:-1]}_{i}>")
        return tokens
    tokens = [placeholder]
    for i in range(1, num_vectors):
        tokens.append(f"{placeholder}_{i}")
    return tokens


def _load_text_encoder(
    base_model: str,
    instance_dir: Path,
    enable_cpa: bool,
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = TextModel.from_pretrained(base_model, subfolder="text_encoder")

    text_encoder_dir = instance_dir / "text_encoder"
    if text_encoder_dir.exists():
        text_encoder.load_adapter(str(text_encoder_dir))
        if enable_cpa:
            text_encoder.set_adapter_mask()
            text_encoder.replace_lora_forward(verbose=False)

    embedding_file = instance_dir / "learned_embeds.bin"
    if not embedding_file.exists():
        raise FileNotFoundError(f"Missing learned embeddings: {embedding_file}")
    learned = torch.load(embedding_file, map_location="cpu")

    placeholder_ids: list[int] = []
    for placeholder, embedding in learned.items():
        load_new_token(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            placeholder=placeholder,
            learned_embedding=embedding,
        )
        tokens = _placeholder_tokens(placeholder, int(embedding.shape[0]))
        placeholder_ids.extend(tokenizer.convert_tokens_to_ids(tokens))

    text_encoder = text_encoder.to(device).eval()
    return tokenizer, text_encoder, set(placeholder_ids)


@torch.inference_mode()
def _prefix_metrics(
    tokenizer,
    text_encoder,
    prompt: str,
    placeholder_ids: set[int],
    device: torch.device,
):
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)

    token_list = input_ids[0].tolist()
    first_placeholder = next(
        (idx for idx, tok in enumerate(token_list) if tok in placeholder_ids),
        None,
    )
    if first_placeholder is None or first_placeholder <= 1:
        return input_ids, [], None
    prefix_indices = list(range(1, first_placeholder))  # skip BOS token

    outputs = text_encoder(input_ids=input_ids)
    hidden = (
        outputs.last_hidden_state
        if hasattr(outputs, "last_hidden_state")
        else outputs[0]
    )
    return input_ids, prefix_indices, hidden


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.annotations_file, "r") as f:
        metadata = json.load(f)

    instances = (
        args.instances if args.instances is not None else sorted(metadata.keys())
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for instance in instances:
        instance_class = metadata[instance].get("class", instance)
        cpa_instance_dir = Path(args.cpa_path) / instance
        naive_instance_dir = Path(args.naive_path) / instance

        if not cpa_instance_dir.exists() or not naive_instance_dir.exists():
            print(f"Skip {instance}: missing cpa or naive checkpoint directory")
            continue

        cpa_tokenizer, cpa_encoder, cpa_placeholder_ids = _load_text_encoder(
            args.base_model,
            cpa_instance_dir,
            enable_cpa=True,
            device=device,
        )
        naive_tokenizer, naive_encoder, naive_placeholder_ids = _load_text_encoder(
            args.base_model,
            naive_instance_dir,
            enable_cpa=False,
            device=device,
        )
        ref_tokenizer, ref_encoder, ref_placeholder_ids = _load_text_encoder(
            args.base_model,
            cpa_instance_dir,
            enable_cpa=False,
            device=device,
        )

        identifier = f"<{instance}>"
        for prompt_template in args.prompts:
            prompt = prompt_template.replace("<INSTANCE>", identifier).replace(
                "SUBJECT", instance_class
            )

            _, ref_prefix_indices, ref_hidden = _prefix_metrics(
                ref_tokenizer,
                ref_encoder,
                prompt,
                ref_placeholder_ids,
                device,
            )
            if ref_hidden is None or not ref_prefix_indices:
                continue

            _, cpa_prefix_indices, cpa_hidden = _prefix_metrics(
                cpa_tokenizer,
                cpa_encoder,
                prompt,
                cpa_placeholder_ids,
                device,
            )
            _, naive_prefix_indices, naive_hidden = _prefix_metrics(
                naive_tokenizer,
                naive_encoder,
                prompt,
                naive_placeholder_ids,
                device,
            )
            if cpa_hidden is None or naive_hidden is None:
                continue

            prefix_count = min(
                len(ref_prefix_indices),
                len(cpa_prefix_indices),
                len(naive_prefix_indices),
            )
            prefix_idx = ref_prefix_indices[:prefix_count]

            ref_tokens = ref_hidden[0, prefix_idx]
            cpa_tokens = cpa_hidden[0, prefix_idx]
            naive_tokens = naive_hidden[0, prefix_idx]

            cpa_cos = F.cosine_similarity(ref_tokens, cpa_tokens, dim=-1).mean().item()
            naive_cos = (
                F.cosine_similarity(ref_tokens, naive_tokens, dim=-1).mean().item()
            )
            cpa_l2 = torch.norm(ref_tokens - cpa_tokens, dim=-1).mean().item()
            naive_l2 = torch.norm(ref_tokens - naive_tokens, dim=-1).mean().item()

            rows.append(
                {
                    "instance": instance,
                    "prompt": prompt,
                    "prefix_token_count": prefix_count,
                    "cpa_prefix_cosine": cpa_cos,
                    "naive_prefix_cosine": naive_cos,
                    "cpa_prefix_l2": cpa_l2,
                    "naive_prefix_l2": naive_l2,
                    "delta_cosine_naive_minus_cpa": naive_cos - cpa_cos,
                    "delta_l2_naive_minus_cpa": naive_l2 - cpa_l2,
                }
            )

    fieldnames = [
        "instance",
        "prompt",
        "prefix_token_count",
        "cpa_prefix_cosine",
        "naive_prefix_cosine",
        "cpa_prefix_l2",
        "naive_prefix_l2",
        "delta_cosine_naive_minus_cpa",
        "delta_l2_naive_minus_cpa",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        print("No rows computed.")
        return

    cpa_cos = sum(r["cpa_prefix_cosine"] for r in rows) / len(rows)
    naive_cos = sum(r["naive_prefix_cosine"] for r in rows) / len(rows)
    cpa_l2 = sum(r["cpa_prefix_l2"] for r in rows) / len(rows)
    naive_l2 = sum(r["naive_prefix_l2"] for r in rows) / len(rows)

    print(f"Saved prefix drift report to {output_csv}")
    print(f"CPA   mean prefix cosine: {cpa_cos:.6f}")
    print(f"Naive mean prefix cosine: {naive_cos:.6f}")
    print(f"CPA   mean prefix L2    : {cpa_l2:.6f}")
    print(f"Naive mean prefix L2    : {naive_l2:.6f}")


if __name__ == "__main__":
    main()
