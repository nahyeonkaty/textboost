#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from textboost.constants import DIFFUSERS_MODEL_DICT
from textboost.utils import find_free_port

PYTHON = sys.executable or "python3"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run TextBoost experiment")
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory where experiment folders are saved.",
    )
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="datasets/dreambooth_n1.json",
        help="Path to annotations JSON file (takes precedence over individual data directories)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="sd21base",
        choices=[
            "sd14",
            "sd15",
            "sd21",
            "sd21base",
        ],
    )
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--desc", type=str, default=None)
    # Training arguments.
    parser.add_argument("--total_steps", type=int, default=100)
    parser.add_argument("--emb_lr", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=2)
    # TextBoost specific arguments.
    parser.add_argument("--lora_rank", type=int, default=1)
    parser.add_argument("--unet_lora_rank", type=int, default=0)
    parser.add_argument(
        "--generator_finetune",
        type=str,
        default=None,
        choices=["none", "lora", "kv", "full"],
    )
    parser.add_argument("--generator_lora_rank", type=int, default=4)
    parser.add_argument(
        "--unet_adapter_mode",
        type=str,
        default="auto",
        choices=["auto", "lora_kv", "full_kv"],
    )
    parser.add_argument("--module", type=str, nargs="+", default=None)
    parser.add_argument("--template", type=str, default="imagenet_small")
    parser.add_argument(
        "--identifier_style",
        type=str,
        default="custom",
        choices=["custom", "ti"],
    )
    parser.add_argument("--disable_cpa", action="store_true")
    parser.add_argument("--expand", action="store_true")
    parser.add_argument(
        "--expand_backend",
        type=str,
        default="legacy_unet",
        choices=["legacy_unet", "text_encoder_bank"],
    )
    parser.add_argument("--expand_layer_indices", type=int, nargs="+", default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--eval_checkpoint", type=int, nargs="*", default=None)
    parser.add_argument("--eval_desc", type=str, default=None)
    parser.add_argument(
        "--eval_outdir",
        type=str,
        default="./images",
        help="Output root directory for generated evaluation images.",
    )
    parser.add_argument("--eval_token_format", type=str, default="<INSTANCE> SUBJECT")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--metrics", type=str, nargs="+", default=["vqa", "dino"])
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    with open(args.annotations_file, "r") as file:
        INSTANCES = json.load(file)

    if args.instances == "none" or args.instances == ["none"]:
        instances = {}
    elif args.instances is not None:
        instances = {}
        for k, v in INSTANCES.items():
            if k in args.instances:
                instances[k] = v
    else:
        instances = INSTANCES

    outdir = str(Path(args.output_root) / f"tb-{args.model}")
    if args.desc is not None:
        outdir += f"-{args.desc}"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    args.model = DIFFUSERS_MODEL_DICT[args.model]
    if args.model == "sd21":
        resolution = 768
    else:
        resolution = 512

    # Run training.
    num_gpu = len(args.gpu.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    accelerate_cmd = [
        PYTHON,
        "-m",
        "accelerate.commands.launch",
        "--mixed_precision=bf16",
        "--num_processes=1",
        "--num_machines=1",
        "--dynamo_backend=no",
        f"--main_process_port={find_free_port()}",
    ]

    train_failures: list[tuple[str, int, list[str]]] = []
    eval_failures: list[tuple[str, int, list[str]]] = []

    if not args.skip_train:
        for name, instance_data in instances.items():
            init_token = instance_data.get(
                "init_token", instance_data.get("initialization")
            )
            if init_token is None:
                init_token = instance_data.get("class", name)
            class_token = (
                instance_data.get("class")
                or instance_data.get("class_token")
                or instance_data.get("instance")
                or name
            )

            cmd = [
                "scripts/train_sd.py",
                f"--pretrained_model_name_or_path={args.model}",
                f"--output_dir={str(Path(outdir) / name)}",
                f"--annotations_file={args.annotations_file}",
                f"--instance={name}",
                f"--class_token={class_token}",
                f"--validation_steps={args.total_steps // 5}",
                # f"--validation_steps={1}",
                f"--placeholder_token=<{name}>",  # Name of the token
                f"--initializer_token={init_token}",
                "--validation_prompts",
                "{}",
                "a {} in the jungle",
                "a {} in the snow",
                "painting of a {} in the Monet style",
                f"--resolution={resolution}",
                f"--lora_rank={args.lora_rank}",
                f"--unet_lora_rank={args.unet_lora_rank}",
                f"--unet_adapter_mode={args.unet_adapter_mode}",
                f"--learning_rate={args.lr}",
                f"--emb_learning_rate={args.emb_lr}",
                # "--emb_lr_scheduler=cosine",
                f"--train_batch_size={args.batch_size // num_gpu}",
                f"--max_train_steps={args.total_steps}",
                f"--checkpointing_steps={args.total_steps // 5}",
                f"--gradient_accumulation_steps={args.accum}",
                f"--template={args.template}",
                f"--identifier_style={args.identifier_style}",
                "--seed=42",
            ]
            if args.generator_finetune is not None:
                cmd.append(f"--generator_finetune={args.generator_finetune}")
            if (
                args.generator_finetune is not None
                and args.generator_lora_rank is not None
            ):
                cmd.append(f"--generator_lora_rank={args.generator_lora_rank}")

            if args.disable_cpa:
                cmd.append("--disable_cpa")
            if args.expand:
                cmd.append("--expand")
                cmd.append(f"--expand_backend={args.expand_backend}")
            if args.expand_layer_indices is not None:
                cmd.extend(
                    ["--expand_layer_indices", *map(str, args.expand_layer_indices)]
                )
            if args.module is not None:
                cmd.extend(["--lora_target_modules", *args.module])

            # Save cmd as text file.
            instance_dir = Path(outdir) / name
            instance_dir.mkdir(parents=True, exist_ok=True)
            with open(instance_dir / "cmd.txt", "w") as file:
                for line in cmd:
                    file.write(f"{line} \\\n")
            result = subprocess.run(accelerate_cmd + cmd)
            if result.returncode != 0:
                train_failures.append((name, result.returncode, accelerate_cmd + cmd))
                print(
                    f"[train][FAIL] instance={name} exit={result.returncode}. Continuing."
                )
            else:
                print(f"[train][OK] instance={name}")

    # Run evaluation.
    if not args.skip_eval:
        if args.eval_checkpoint is None:
            checkpointing_steps = list(
                range(args.total_steps, 0, -(args.total_steps // 5))
            )
        elif len(args.eval_checkpoint) == 0:
            checkpointing_steps = [None]
        else:
            checkpointing_steps = args.eval_checkpoint

        for ckpt in checkpointing_steps:
            cmd = [
                PYTHON,
                "scripts/evaluate.py",
                outdir,
                f"--train_data={args.annotations_file}",
                "--outdir",
                args.eval_outdir,
                "--token_format",
                args.eval_token_format,
                "--batch_size=8",
                "--guidance_scale",
                str(args.guidance_scale),
                "--num_inference_steps",
                str(args.num_inference_steps),
                "--metric",
                *args.metrics,
                "--seeds",
                *map(str, args.seeds),
            ]
            if ckpt is not None:
                cmd.append(f"--checkpoint={ckpt}")
            if args.eval_desc is not None:
                cmd.extend(["--desc", args.eval_desc])
            if args.disable_cpa:
                cmd.append("--disable_cpa")
            if args.instances is not None:
                cmd.extend(["--instances", *args.instances])
            result = subprocess.run(cmd)
            if result.returncode != 0:
                ckpt_name = "final" if ckpt is None else str(ckpt)
                eval_failures.append((ckpt_name, result.returncode, cmd))
                print(
                    f"[eval][FAIL] checkpoint={ckpt_name} exit={result.returncode}. Continuing."
                )
            else:
                ckpt_name = "final" if ckpt is None else str(ckpt)
                print(f"[eval][OK] checkpoint={ckpt_name}")

    if train_failures or eval_failures:
        failure_log = Path(outdir) / "failures.log"
        with open(failure_log, "w") as f:
            if train_failures:
                f.write("[train]\n")
                for name, code, cmd in train_failures:
                    f.write(f"instance={name} exit={code}\n")
                    f.write("cmd: " + " ".join(cmd) + "\n")
            if eval_failures:
                f.write("[eval]\n")
                for ckpt_name, code, cmd in eval_failures:
                    f.write(f"checkpoint={ckpt_name} exit={code}\n")
                    f.write("cmd: " + " ".join(cmd) + "\n")
        print(
            f"Completed with failures. train_failures={len(train_failures)}, "
            f"eval_failures={len(eval_failures)}. See {failure_log}"
        )


if __name__ == "__main__":
    main()
