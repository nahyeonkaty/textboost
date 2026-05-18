#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from textboost.constants import DIFFUSERS_MODEL_DICT
from textboost.evaluation.dreambooth import (
    INSTANCES,
    LIVE_PROMPTS,
    OBJ_PROMPTS,
    is_live,
)
from textboost.pipelines.loaders import (
    load_sana_pipeline,
    load_sd_pipeline,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to model")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument(
        "--token_format",
        type=str,
        default="<INSTANCE> SUBJECT",
        help=(
            "Token format for the prompt "
            "[sks SUBJECT] for DreamBooth models, "
            "[<INSTANCE>] for Textual Inversion models, "
            "[<INSTANCE> SUBJECT] for CustomDiffusion and TextBoost."
        ),
    )
    parser.add_argument("--train_data", type=str, default="./datasets/dreambooth.json")
    parser.add_argument("--dreambooth_dir", type=str, default="./datasets/dreambooth")
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="datasets/dreambooth_mask",
        help=(
            "Mask directory for DINO reference masking. "
            "When this directory exists, metric CSV names get '-mask' suffix."
        ),
    )

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--outdir", type=str, default="./images")
    parser.add_argument("--desc", type=str, default=None)

    # Generation parameters.
    parser.add_argument("--skip_gen", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument(
        "--disable_cpa",
        action="store_true",
        help="Disable CPA masking when loading text-encoder adapters (for naive baselines).",
    )

    # Evaluation parameters.
    parser.add_argument("--metric", type=str, nargs="+", default=["vqa", "dino"])
    return parser.parse_args()


def _sanitize_filename_component(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return value or "default"


def _mask_metric_desc(desc: str | None, mask_dir: str | None) -> str | None:
    if mask_dir is None or not Path(mask_dir).exists():
        return desc
    if desc is None:
        return "mask"
    if desc.endswith("-mask") or desc.endswith("_mask"):
        return desc
    return f"{desc}-mask"


def _resolve_model_name(model_path, pretrained_model):
    if isinstance(pretrained_model, str) and pretrained_model.lower().startswith(
        "flux"
    ):
        return "flux"
    if pretrained_model in {"sdxl", "sdxlbase"}:
        return DIFFUSERS_MODEL_DICT["sdxlbase"]
    if pretrained_model in DIFFUSERS_MODEL_DICT:
        return DIFFUSERS_MODEL_DICT[pretrained_model]
    if pretrained_model in DIFFUSERS_MODEL_DICT.values():
        return pretrained_model

    path_str = str(model_path).lower()
    if "sana" in path_str and "4.8b" in path_str:
        return DIFFUSERS_MODEL_DICT["sana1.5_4.8b"]
    if "sana" in path_str:
        return DIFFUSERS_MODEL_DICT["sana1.5_1.6b"]
    if "sd21base" in path_str:
        return DIFFUSERS_MODEL_DICT["sd21base"]
    if "sdxl" in path_str:
        return DIFFUSERS_MODEL_DICT["sdxlbase"]
    if "sd21" in path_str:
        return DIFFUSERS_MODEL_DICT["sd21"]
    if "sd15" in path_str:
        return DIFFUSERS_MODEL_DICT["sd15"]
    if "sd14" in path_str:
        return DIFFUSERS_MODEL_DICT["sd14"]
    raise ValueError(f"Cannot resolve model type from path: {model_path}")


def _is_sana_model(model_name: str) -> bool:
    return "sana" in model_name.lower()


def load_pipeline(
    model_path,
    pretrained_model,
    dtype=torch.bfloat16,
    enable_cpa_mask: bool = True,
):
    print(model_path, pretrained_model)
    model = _resolve_model_name(model_path, pretrained_model)
    if _is_sana_model(model):
        pipeline, identifiers = load_sana_pipeline(
            model_name=model,
            checkpoint_path=model_path,
        )
    else:
        pipeline, identifiers = load_sd_pipeline(
            model=model,
            checkpoint_path=model_path,
            enable_cpa_mask=enable_cpa_mask,
        )

    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.eval().requires_grad_(False)
    pipeline.text_encoder.eval().requires_grad_(False)
    if hasattr(pipeline, "transformer"):
        pipeline.transformer.eval().requires_grad_(False)
    if hasattr(pipeline, "unet"):
        pipeline.unet.eval().requires_grad_(False)
    pipeline = pipeline.to(dtype=dtype)

    torch.cuda.empty_cache()
    return pipeline, identifiers


def generate_from_pipeline(
    pipeline,
    instance,
    identifier,
    seeds,
    outdir,
    batch_size=8,
    num_inference_steps=25,
    guidance_scale=7.5,
    device="cuda",
):
    assert instance in INSTANCES, f"Invalid instance: {instance}"
    prompt_list = LIVE_PROMPTS if is_live(instance) else OBJ_PROMPTS

    if outdir.endswith("/"):
        outdir = outdir[:-1]

    cls = INSTANCES[instance]

    print(f"Identifiers: {identifier}")
    print(f"Seeds: {seeds}")

    # Create prompt-seed pairs.
    all_prompts = []
    all_seeds = []
    for prompt_template in prompt_list:
        for seed in seeds:
            all_prompts.append(prompt_template.format(identifier))
            all_seeds.append(seed)
    print(f"{len(prompt_list)} x {len(seeds)} = {len(all_prompts)}")

    # Sample prompts in batches with batch_size.
    for batch_start in tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[batch_start : batch_start + batch_size]
        batch_seeds = all_seeds[batch_start : batch_start + batch_size]
        batch_generators = [
            torch.Generator(device=device).manual_seed(seed) for seed in batch_seeds
        ]
        # print(batch_prompts, batch_seeds)

        inference_kwargs = dict(
            prompt=batch_prompts,
            generator=batch_generators,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        images = pipeline(**inference_kwargs).images

        for prompt, seed, image in zip(batch_prompts, batch_seeds, images):
            dst = Path(outdir) / f"seed{seed}" / instance
            dst.mkdir(parents=True, exist_ok=True)
            filename = f"{prompt.replace(identifier, cls).replace(' ', '_')}.png"
            image.save(dst / filename)
    del pipeline


def generate(args, device):
    if args.instances is not None:
        instances = {}
        for name, cls in INSTANCES.items():
            if name in args.instances:
                instances[name] = cls
    else:
        instances = INSTANCES

        subdirs = list(Path(args.path).iterdir())
        subdirs = list(filter(lambda x: x.is_dir(), subdirs))
        # for instance in INSTANCES.keys():
        #     assert instance in subdirs, f"Missing instance: {instance}"
        # assert len(subdirs) == 30, f"Invalid number of instances: {len(subdirs)}"

    if args.outdir.endswith("/"):
        args.outdir = args.outdir[:-1]
    if args.path.endswith("/"):
        args.path = args.path[:-1]

    if args.checkpoint is not None:
        basename = f"{Path(args.path).name}-{args.checkpoint}"
    else:
        basename = Path(args.path).name
    outdir = Path(args.outdir) / basename
    if args.desc is not None:
        outdir = outdir.parent / f"{outdir.name}_{args.desc}"
    if args.skip_gen:
        return str(outdir)

    if args.model is not None:
        model = args.model
    elif "flux" in args.path.lower():
        model = "flux"
    elif "sd21base" in args.path:
        model = "sd21base"
    elif "sdxl" in args.path.lower():
        model = "sdxlbase"
    elif "sd21" in args.path:
        model = "sd21"
    elif "sd15" in args.path:
        model = "sd15"
    elif "sd14" in args.path:
        model = "sd14"
    elif "4.8b" in args.path:
        model = "sana1.5_4.8b"
    else:
        model = "sana1.5_1.6b"

    if isinstance(model, str) and model.lower().startswith("flux"):
        print(
            "FLUX checkpoint loading is not implemented in scripts/evaluate.py yet; skipping generation."
        )
        return str(outdir)

    failed_instances: list[tuple[str, str]] = []
    completed_instances: list[str] = []

    for instance in tqdm(instances):
        ckpt = f"checkpoint-{args.checkpoint}" if args.checkpoint is not None else ""
        model_path = Path(args.path) / instance / ckpt
        if not model_path.exists():
            print(f"Skip missing checkpoint path: {model_path}")
            continue

        try:
            pipeline, identifiers = load_pipeline(
                model_path,
                model,
                enable_cpa_mask=not args.disable_cpa,
            )
            pipeline = pipeline.to(device)

            learned_identifier = identifiers[0] if identifiers else f"<{instance}>"
            identifier = args.token_format.replace("<INSTANCE>", learned_identifier)
            identifier = identifier.replace("INSTANCE", learned_identifier.strip("<>"))
            identifier = identifier.replace("SUBJECT", INSTANCES[instance])

            generate_from_pipeline(
                pipeline=pipeline,
                instance=instance,
                identifier=identifier,
                seeds=args.seeds,
                outdir=str(outdir),
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                device=device,
            )
            completed_instances.append(instance)
        except Exception as e:
            failed_instances.append((instance, repr(e)))
            print(f"[gen][FAIL] instance={instance}: {e}")
            torch.cuda.empty_cache()

    if failed_instances:
        print(
            f"[gen] completed={len(completed_instances)}, failed={len(failed_instances)}"
        )
        for instance, reason in failed_instances:
            print(f"  - {instance}: {reason}")
    else:
        print(f"[gen] completed={len(completed_instances)}, failed=0")
    return str(outdir)


@torch.inference_mode()
def main():
    args = parse_arguments()

    if args.path.endswith("/"):
        args.path = args.path[:-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generated_image_path = generate(args, device)
    with open(args.train_data, "r") as file:
        data = json.load(file)

    requested_metrics = [m for m in args.metric if m in {"dino", "vqa"}]
    unsupported_metrics = [m for m in args.metric if m not in {"dino", "vqa"}]
    if unsupported_metrics:
        print(f"Skipping unsupported metrics: {unsupported_metrics}")

    score_dict = {}
    if requested_metrics:
        try:
            from textboost.evaluation.metrics import EvaluationConfig, EvaluationSuite
        except Exception as exc:
            print(
                "Metric evaluation dependencies are unavailable; generation completed."
            )
            print(f"Reason: {exc}")
        else:
            instances = (
                args.instances if args.instances is not None else list(data.keys())
            )
            evaluator = EvaluationSuite(
                EvaluationConfig(
                    device=str(device),
                    batch_size=args.batch_size,
                    verbose=True,
                )
            )
            try:
                results = evaluator.evaluate(
                    generated_dir=generated_image_path,
                    reference_dir=args.dreambooth_dir,
                    instances=instances,
                    mask_dir=args.mask_dir,
                    metrics=requested_metrics,
                )
            finally:
                evaluator.cleanup()

            if "dino" in results and len(results["dino"]) > 0:
                score_dict["dino"] = results["dino"]
            if "vqa" in results and len(results["vqa"]) > 0:
                score_dict["vqa"] = results["vqa"]

    if not score_dict:
        print("No metrics were computed.")
        return

    # Save scores to file.
    # Use per-condition metric files when --desc is set to avoid mixing rows.
    metric_desc = _mask_metric_desc(args.desc, args.mask_dir)
    if metric_desc:
        csvfile = (
            Path(args.path) / f"metric_{_sanitize_filename_component(metric_desc)}.csv"
        )
    else:
        csvfile = Path(args.path) / "metric.csv"
    headline = ["checkpoint", "dino", "vqa"]
    if not csvfile.exists():
        with open(csvfile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(headline)
    print(headline)
    with open(csvfile, "a") as f:
        writer = csv.writer(f)
        checkpoint_name = args.checkpoint if args.checkpoint is not None else "final"
        dino_score = str(np.mean(score_dict["dino"])) if "dino" in score_dict else ""
        vqa_score = str(np.mean(score_dict["vqa"])) if "vqa" in score_dict else ""
        line = [checkpoint_name, dino_score, vqa_score]
        writer.writerow(line)
        print(line)
    print(f"Saved metrics to: {csvfile}")


if __name__ == "__main__":
    main()
