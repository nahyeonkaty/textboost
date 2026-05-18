#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

from textboost.constants import DIFFUSERS_MODEL_DICT
from textboost.utils import find_free_port


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run TextBoost experiment")
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="datasets/qwen_annotations.json",
        help="Path to annotations JSON file (takes precedence over individual data directories)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="sana1.5_1.6b",
        choices=[
            "sana600m512",
            "sana600m1024",
            "sana1600m512",
            "sana1600m1024",
            "sana1.5_1.6b",
            "sana1.5_4.8b",
        ],
    )
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--desc", type=str, default=None)
    # Training arguments.
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--emb_lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=2)
    # TextBoost specific arguments.
    parser.add_argument("--lora_rank", type=int, default=1)
    parser.add_argument("--module", type=str, nargs="+", default=None)
    parser.add_argument("--expand", action="store_true")
    parser.add_argument(
        "--generator_finetune",
        type=str,
        default="none",
        choices=["none", "lora", "kv", "full"],
    )
    parser.add_argument("--generator_lora_rank", type=int, default=4)
    parser.add_argument("--generator_target_modules", type=str, nargs="+", default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    with open(args.annotations_file, "r") as file:
        data = json.load(file)

    if args.instances == "none":  # skip training, run evaluation directly.
        instances = {}
    elif args.instances is not None:
        instances = {}
        for k, v in data.items():
            if k in args.instances:
                instances[k] = v
    else:
        instances = data

    outdir = f"outputs/tb-{args.model}"
    if args.desc is not None:
        outdir += f"-{args.desc}"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    args.model = DIFFUSERS_MODEL_DICT[args.model]
    if "512" in args.model:
        resolution = 512
    else:
        resolution = 1024

    # Run training.
    num_gpu = len(args.gpu.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    accelerate_cmd = [
        "accelerate",
        "launch",
        "--mixed_precision=bf16",
        "--num_processes=1",
        "--num_machines=1",
        "--dynamo_backend=no",
        f"--main_process_port={find_free_port()}",
    ]

    for name, data in instances.items():
        init_token = data["init_token"]

        id_style = "ti"

        cmd = [
            "scripts/train_sana.py",
            f"--pretrained_model_name_or_path={args.model}",
            f"--output_dir=./{outdir}/{name}",
            f"--annotations_file={args.annotations_file}",
            f"--instance={name}",
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
            f"--generator_finetune={args.generator_finetune}",
            f"--generator_lora_rank={args.generator_lora_rank}",
            f"--learning_rate={args.lr}",
            f"--emb_learning_rate={args.emb_lr}",
            # "--emb_lr_scheduler=cosine",
            f"--train_batch_size={args.batch_size // num_gpu}",
            f"--max_train_steps={args.total_steps}",
            f"--checkpointing_steps={args.total_steps // 5}",
            f"--gradient_accumulation_steps={args.accum}",
            "--template=imagenet_small",
            f"--identifier_style={id_style}",
            "--seed=42",
        ]

        if args.expand:
            cmd.append("--expand")
        if args.module is not None:
            cmd.extend(["--target_modules", *args.module])
        if args.generator_target_modules is not None:
            cmd.extend(["--generator_target_modules", *args.generator_target_modules])

        # Save cmd as text file.
        instance_dir = Path(f"{outdir}/{name}")
        instance_dir.mkdir(parents=True, exist_ok=True)
        with open(instance_dir / "cmd.txt", "w") as file:
            for line in cmd:
                file.write(f"{line} \\\n")
        subprocess.run(accelerate_cmd + cmd)

    # Run evaluation.
    checkpointing_steps = range(args.total_steps, 0, -(args.total_steps // 5))
    for ckpt in checkpointing_steps:
        cmd = [
            "python",
            "scripts/evaluate.py",
            outdir,
            f"--checkpoint={ckpt}",
            f"--train_data={args.annotations_file}",
            "--token_format",
            "<INSTANCE> SUBJECT",
            "--batch_size=8",
        ]
        if args.instances is not None:
            cmd.extend(["--instances", *args.instances])
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
