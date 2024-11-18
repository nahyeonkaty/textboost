#!/usr/bin/env python3
import argparse
import os
import subprocess

# Classes - ("subject_name", "class", "init token"),
INSTANCES = [
    ("00", "A seascape and cliffs in {}", "watercolor painting style"),  # DCO
    ("01", "A house in {}", "watercolor painting style"),
    ("02", "A cat in {}", "watercolor painting style"),
    # ("03", "Flowers in {}", "watercolor painting style"),
    ("03", "Row of flowers in {}", "watercolor painting style"),  # DCO
    ("04", "A village in {}", "oil painting style"),
    ("05", "A village in {}", "line drawing style"),
    # ("06", "A portrait of a man in {}", "line drawing style"),
    ("07", "A portrait of a person wearing a hat in {}", "oil painting style"),
    ("08", "A woman walking a dog in {}", "flat cartoon illustration style"),
    ("09", "A woman working on a laptop in {}", "flat cartoon illustration style"),
    ("10", "A Christmas tree in {}", "sticker style"),
    ("11", "A wave in {}", "abstract rainbow colored flowing smoke wave design"),
    ("12", "A mushroom in {}", "glowing style"),
    # ("13", "a cat sits in front of a window in {}", None),
    # ("14", "a path through the woods with trees and fog in {}", None),
    ("15", "Slices of watermelon and clouds in the background in {}", "3D rendering style"),
    ("16", "A house in {}", "3D rendering style"),
    ("17", "A thumbs up in {}", "glowing style"),
    # ("18", "A woman in {}", "3D rendering style"),
    ("18", "A female figure with exaggerated proportions in {}", "modern 3D rendering style"),  # DCO
    ("19", "A bear in {} animal", "kid crayon drawing style"),
    # ("20", "a statue of a man's head in {}", "silver sculpture style"),
    ("21", "A flower in {}", "melting golden 3D rendering style"),
    ("22", "A Viking face with beard in {}", "wooden sculpture style"),
]

parser = argparse.ArgumentParser(description='Run TextBoost experiment')
parser.add_argument("-g", "--gpu", type=str, default="7")
parser.add_argument("-m", "--model", type=str, default="sd2.1")
parser.add_argument("--instances", type=str, nargs="+", default=None)
parser.add_argument("--augment", type=str, default="pda")
parser.add_argument("--lora-rank", type=int, default=4)
parser.add_argument("--null-prob", type=float, default=0.1)
parser.add_argument("--kpl-weight", type=float, default=0.1)
parser.add_argument("--no-weighted-sample", action="store_true", default=False)
parser.add_argument("--no-inversion", action="store_true", default=False)
parser.add_argument("--desc", type=str, default=None)


def main(args):
    if args.instances is not None:
        instances = []
        for name, cls in INSTANCES:
            if name in args.instances:
                instances.append((name, cls))
    else:
        instances = INSTANCES

    outdir = f"output/tb_style-{args.model}"
    if args.desc is not None:
        outdir += f"-{args.desc}"

    os.makedirs(outdir, exist_ok=True)

    model = args.model.lower().replace("-", "")
    if model == "sd1.4":
        args.model = "CompVis/stable-diffusion-v1-4"
    elif model == "sd1.5":
        # args.model = "runwayml/stable-diffusion-v1-5"
        args.model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    elif model == "sd2.1":
        args.model = "stabilityai/stable-diffusion-2-1"

    num_gpu = len(args.gpu.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torchrun_cmd = [
        "torchrun",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        f"--nproc-per-node={num_gpu}",
    ]
    for name, template, init_token in instances:
        cmd = [
            "train_textboost.py",
            f"--pretrained_model_name_or_path={args.model}",
            f"--instance_data_dir=./datasets/styledrop/{name}",
            f"--output_dir=./{outdir}/{name}",
            "--instance_token=<0>",
            f"--validation_prompts",
            "a man in <0>",
            "a cat in <0>",
            "flowers in <0>",
            "a dog in <0>",
            "--validation_steps=25",
            "--placeholder_token", f"<{name}>",
            "--initializer_token", f"{init_token}",
            f"--lora_rank={args.lora_rank}",
            "--learning_rate=1e-4",
            "--emb_learning_rate=1e-3",
            "--train_batch_size=4",
            "--max_train_steps=150",
            "--checkpointing_steps=25",
            "--gradient_accumulation_steps=1",
            f"--augment={args.augment}",
            f"--kpl_weight={args.kpl_weight}",
            f"--null_prob={args.null_prob}",
            f"--template={template}",
            "--augment_ops=style",
            "--mixing",
            # "--mixed_precision=fp16",
        ]
        if not args.no_inversion:
            cmd.append("--augment_inversion")
        if args.no_weighted_sample:
            cmd.append("--disable_weighted_sample")
        subprocess.run(torchrun_cmd + cmd)

        # save cmd as text file
        cmd_txt = "\n".join(cmd)
        with open(f"{outdir}/{name}/cmd.txt", "w") as file:
            file.write(cmd_txt)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
