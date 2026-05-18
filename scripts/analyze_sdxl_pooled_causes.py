#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer

from textboost.datasets import (
    IMAGENET_STYLE_TEMPLATES_SMALL,
    IMAGENET_TEMPLATES_SMALL,
    TEXTBOOST_TEMPLATES,
)
from textboost.text_encoders.clip import TextModelWithProjection
from textboost.ti_utils import add_new_token, load_new_token


TEMPLATE_MAP = {
    "imagenet_small": IMAGENET_TEMPLATES_SMALL,
    "imagenet_style_small": IMAGENET_STYLE_TEMPLATES_SMALL,
    "textboost": TEXTBOOST_TEMPLATES,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SDXL pooled embedding changes by decomposing effects from "
            "learned token embeddings vs text_encoder_2 adapters."
        )
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Optional prompt templates/prompts. Supports '{}' and '<*>' placeholders.",
    )
    parser.add_argument(
        "--use_template_prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true and --prompts not provided, use full template set from args.json template.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Default: <experiment_dir>/<instance>/pooled_embeddings/pooled_cause_analysis.csv",
    )
    parser.add_argument(
        "--enable_cpa_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply CPA adapter masking/forward replacement when adapter exists.",
    )
    return parser.parse_args()


def _is_adapter_checkpoint_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    names = {p.name for p in path.iterdir()}
    if "adapter_config.json" in names:
        return True
    return any(
        name in {"adapter_model.bin", "adapter_model.safetensors"} for name in names
    )


def _format_prompt(template: str, identifier_text: str) -> str:
    if "{}" in template:
        return template.format(identifier_text)
    if "<*>" in template:
        return template.replace("<*>", identifier_text)
    return template


def _build_adapter_mask(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    adapter_mask = attention_mask.clone()
    unknown_tokens = (input_ids > 49407).float()
    first_unknown_token = torch.argmax(unknown_tokens, dim=1)
    for i, first_idx in enumerate(first_unknown_token.tolist()):
        adapter_mask[i, :first_idx] = 0.0
    return adapter_mask


@torch.inference_mode()
def _encode_pooled(
    text_encoder_2,
    tokenizer_2,
    prompts: list[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    pooled_batches = []
    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i : i + batch_size]
        text_inputs = tokenizer_2(
            prompt_batch,
            truncation=True,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        adapter_mask = _build_adapter_mask(input_ids, attention_mask)
        out = text_encoder_2(
            input_ids,
            output_hidden_states=True,
            adapter_mask=adapter_mask,
        )
        pooled_batches.append(out[0].detach().float().cpu())
    return torch.cat(pooled_batches, dim=0)


def _load_model_condition(
    model_name: str,
    placeholder: str,
    initializer_token: str,
    learned_embedding: torch.Tensor,
    use_learned_embedding: bool,
    adapter_dir: Path,
    use_adapter: bool,
    enable_cpa_mask: bool,
    device: torch.device,
):
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    text_encoder_2 = TextModelWithProjection.from_pretrained(
        model_name, subfolder="text_encoder_2"
    )

    if use_learned_embedding:
        identifier = load_new_token(
            text_encoder=text_encoder_2,
            tokenizer=tokenizer_2,
            placeholder=placeholder,
            learned_embedding=learned_embedding,
        )
    else:
        new_token = add_new_token(
            tokenizer=tokenizer_2,
            text_encoder=text_encoder_2,
            placeholder=placeholder,
            init_token=initializer_token,
        )
        identifier = new_token.identifier

    adapter_loaded = False
    if use_adapter and _is_adapter_checkpoint_dir(adapter_dir):
        text_encoder_2.load_adapter(str(adapter_dir))
        adapter_loaded = True
        if enable_cpa_mask:
            text_encoder_2.set_adapter_mask()
            text_encoder_2.replace_lora_forward(verbose=False)

    text_encoder_2 = text_encoder_2.to(device).eval()
    return tokenizer_2, text_encoder_2, identifier, adapter_loaded


def _discover_steps(instance_dir: Path) -> list[int]:
    steps = []
    for ckpt in instance_dir.glob("checkpoint-*"):
        if not ckpt.is_dir():
            continue
        try:
            step = int(ckpt.name.split("-")[-1])
        except ValueError:
            continue
        steps.append(step)
    return sorted(set(steps))


def _resolve_prompt_templates(instance_dir: Path, args) -> tuple[list[str], dict]:
    args_json = instance_dir / "args.json"
    if not args_json.exists():
        raise FileNotFoundError(f"Missing args.json: {args_json}")

    with open(args_json, "r") as f:
        train_args = json.load(f)

    if args.prompts is not None:
        templates = args.prompts
    elif args.use_template_prompts:
        template_name = train_args.get("template", "imagenet_small")
        templates = TEMPLATE_MAP.get(template_name, IMAGENET_TEMPLATES_SMALL)
    else:
        templates = train_args.get("validation_prompts") or ["{}"]

    return list(templates), train_args


def main():
    args = parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    experiment_dir = Path(args.experiment_dir)
    instance_dir = experiment_dir / args.instance
    if not instance_dir.exists():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")

    templates, train_args = _resolve_prompt_templates(instance_dir, args)

    placeholder = train_args.get("placeholder_token", f"<{args.instance}>")
    class_token = train_args.get("class_token", args.instance)
    initializer_token = train_args.get("initializer_token", class_token)
    identifier_style = train_args.get("identifier_style", "custom")

    steps = args.steps if args.steps else _discover_steps(instance_dir)
    if not steps:
        raise ValueError(f"No checkpoint-* directories found in {instance_dir}")

    out_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else instance_dir / "pooled_embeddings" / "pooled_cause_analysis.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for step in steps:
        ckpt_dir = instance_dir / f"checkpoint-{step}"
        learned_path = ckpt_dir / "learned_embeds_2.bin"
        adapter_dir = ckpt_dir / "text_encoder_2"
        if not learned_path.exists():
            print(f"[skip] step={step}: missing {learned_path}")
            continue

        learned_dict = torch.load(learned_path, map_location="cpu")
        if placeholder not in learned_dict:
            if len(learned_dict) != 1:
                print(
                    f"[skip] step={step}: placeholder {placeholder} not found in learned_embeds_2 keys={list(learned_dict.keys())}"
                )
                continue
            placeholder_key = next(iter(learned_dict.keys()))
        else:
            placeholder_key = placeholder

        learned_embedding = learned_dict[placeholder_key]

        cond = {}
        for name, use_learned, use_adapter in [
            ("init_no_adapter", False, False),
            ("learned_no_adapter", True, False),
            ("init_with_adapter", False, True),
            ("learned_with_adapter", True, True),
        ]:
            tokenizer_2, text_encoder_2, identifier, adapter_loaded = (
                _load_model_condition(
                    model_name=args.pretrained_model_name_or_path,
                    placeholder=placeholder_key,
                    initializer_token=initializer_token,
                    learned_embedding=learned_embedding,
                    use_learned_embedding=use_learned,
                    adapter_dir=adapter_dir,
                    use_adapter=use_adapter,
                    enable_cpa_mask=args.enable_cpa_mask,
                    device=device,
                )
            )

            identifier_text = identifier
            if identifier_style != "ti" and class_token:
                identifier_text = f"{identifier_text} {class_token}".strip()
            prompts = [_format_prompt(t, identifier_text) for t in templates]

            pooled = _encode_pooled(
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                prompts=prompts,
                device=device,
                batch_size=args.batch_size,
            )
            cond[name] = {
                "pooled": pooled,
                "prompts": prompts,
                "adapter_loaded": adapter_loaded,
            }

            del text_encoder_2
            del tokenizer_2
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # per-prompt decomposition
        for i, prompt in enumerate(cond["learned_with_adapter"]["prompts"]):
            e0 = cond["init_no_adapter"]["pooled"][i]
            et = cond["learned_no_adapter"]["pooled"][i]
            ea = cond["init_with_adapter"]["pooled"][i]
            eb = cond["learned_with_adapter"]["pooled"][i]

            token_vec = et - e0
            adapter_vec = ea - e0
            total_vec = eb - e0
            interaction_vec = eb - et - ea + e0

            rows.append(
                {
                    "instance": args.instance,
                    "step": step,
                    "prompt_idx": i,
                    "prompt": prompt,
                    "adapter_loaded": int(
                        cond["learned_with_adapter"]["adapter_loaded"]
                    ),
                    "l2_total": float(torch.norm(total_vec, p=2).item()),
                    "l2_token_main": float(torch.norm(token_vec, p=2).item()),
                    "l2_adapter_main": float(torch.norm(adapter_vec, p=2).item()),
                    "l2_interaction": float(torch.norm(interaction_vec, p=2).item()),
                    "cos_total_vs_token": float(
                        F.cosine_similarity(
                            total_vec.unsqueeze(0), token_vec.unsqueeze(0), dim=-1
                        ).item()
                    ),
                    "cos_total_vs_adapter": float(
                        F.cosine_similarity(
                            total_vec.unsqueeze(0), adapter_vec.unsqueeze(0), dim=-1
                        ).item()
                    ),
                    "cos_init_to_both": float(
                        F.cosine_similarity(
                            e0.unsqueeze(0), eb.unsqueeze(0), dim=-1
                        ).item()
                    ),
                    "cos_init_to_token_only": float(
                        F.cosine_similarity(
                            e0.unsqueeze(0), et.unsqueeze(0), dim=-1
                        ).item()
                    ),
                    "cos_init_to_adapter_only": float(
                        F.cosine_similarity(
                            e0.unsqueeze(0), ea.unsqueeze(0), dim=-1
                        ).item()
                    ),
                }
            )

    if not rows:
        print("No rows were produced.")
        return

    fieldnames = [
        "instance",
        "step",
        "prompt_idx",
        "prompt",
        "adapter_loaded",
        "l2_total",
        "l2_token_main",
        "l2_adapter_main",
        "l2_interaction",
        "cos_total_vs_token",
        "cos_total_vs_adapter",
        "cos_init_to_both",
        "cos_init_to_token_only",
        "cos_init_to_adapter_only",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # print quick summary by step
    import pandas as pd

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("step", as_index=False)[
            [
                "l2_total",
                "l2_token_main",
                "l2_adapter_main",
                "l2_interaction",
                "cos_init_to_both",
                "cos_init_to_token_only",
                "cos_init_to_adapter_only",
            ]
        ]
        .mean()
        .sort_values("step")
    )

    print(f"Saved: {out_csv}")
    print("\nMean by step:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
