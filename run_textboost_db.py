#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess

# subject_name, class, init_token
INSTANCES = [
    ("backpack", "backpack", "red"),
    ("backpack_dog", "backpack", "character"),
    ("bear_plushie", "stuffed animal", "bear"),
    ("berry_bowl", "bowl", "white"),
    ("can", "can", "beer"),
    ("candle", "candle", "jar"),
    ("cat", "cat", "orange"),
    ("cat2", "cat", "gray"),
    ("clock", "clock", "yellow"),
    ("colorful_sneaker", "sneaker", "color"),
    ("dog", "dog", "corgi"),
    ("dog2", "dog", "fluffy"),
    ("dog3", "dog", "poodle"),
    ("dog5", "dog", "dachshund"),
    ("dog6", "dog", "corgi"),
    ("dog7", "dog", "retriever"),
    ("dog8", "dog", "border collie"),
    ("duck_toy", "toy", "rubber"),
    ("fancy_boot", "boot", "fringe"),
    ("grey_sloth_plushie", "stuffed animal", "sloth"),
    ("monster_toy", "toy", "stuffed"),
    ("pink_sunglasses", "glasses", "pink"),
    ("poop_emoji", "toy", "poop"),
    ("rc_car", "toy", "car"),
    ("red_cartoon", "cartoon", "devil"),
    ("robot_toy", "toy", "robot"),
    ("shiny_sneaker", "sneaker", "rainbow"),
    ("teapot", "teapot", "brown"),
    ("vase", "vase", "red"),
    ("wolf_plushie", "stuffed animal", "dog"),
]

def parse_args():
    parser = argparse.ArgumentParser(description='Run TextBoost experiment')
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-m", "--model", type=str, default="sd21base")
    parser.add_argument("--instances", type=str, nargs="+", default=None)

    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--emb-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--train-params", type=str, default="none")
    parser.add_argument("--augment", type=str, default="pda")
    parser.add_argument("--augment-p", type=float, default=0.5)
    parser.add_argument("--null-prob", type=float, default=0.1)
    parser.add_argument("--kpl-weight", type=float, default=0.1)

    parser.add_argument("--no-weighted-sample", action="store_true", default=False)
    parser.add_argument("--no-inversion", action="store_true", default=False)
    parser.add_argument("--mixing", action="store_true", default=False)

    parser.add_argument("--desc", type=str, default=None)
    args = parser.parse_args()
    return args


def main(args):
    if args.instances is not None:
        instances = []
        for name, cls, init_token in INSTANCES:
            if name in args.instances:
                instances.append((name, cls, init_token))
    else:
        instances = INSTANCES

    num_str = "all" if args.num_samples is None else f"n{args.num_samples}"
    outdir = f"output/tb-{args.model}-{num_str}"
    if args.desc is not None:
        outdir += f"-{args.desc}"

    os.makedirs(outdir, exist_ok=True)

    model = args.model.lower().replace("-", "")
    if model == "sd14":
        args.model = "CompVis/stable-diffusion-v1-4"
    elif model == "sd15":
        args.model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    elif model == "sd21base":
        args.model = "stabilityai/stable-diffusion-2-1-base"
    elif model == "sd21":
        args.model = "stabilityai/stable-diffusion-2-1"

    resolution = {
        "sd14": 512,
        "sd15": 512,
        "sd21base": 512,
        "sd21": 768,
    }[model]

    data_dir = "datasets/dreambooth_n1_train"

    num_gpu = len(args.gpu.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torchrun_cmd = [
        "torchrun",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        f"--nproc-per-node={num_gpu}",
    ]
    for name, cls, init_token in instances:
        # if init_token is None:
        #     init_token = cls
        # identifier = f"<0> {cls}"
        # init_token = cls
        init_token = f"{init_token} {cls}"
        identifier = "<0>"
        cmd = [
            "train_textboost.py",
            f"--pretrained_model_name_or_path={args.model}",
            f"--instance_data_dir={os.path.join(data_dir, name)}",
            f"--output_dir=./{outdir}/{name}",
            f"--class_token={cls}",
            # f"--instance_token=<{name}> {cls}",
            # f"--validation_prompt=a <{name}> {cls} in the jungle",
            f"--instance_token={identifier}",
            f"--validation_steps={args.total_steps // 5}",
            f"--placeholder_token=<{name}>",  # Name of the token
            f"--initializer_token={init_token}",
            f"--validation_prompts",
            f"photo of a {identifier}",
            f"a {identifier} in the jungle",
            f"a {identifier} in the bucket",
            f"painting of a {identifier} in the Monet style",
            f"--resolution={resolution}",
            f"--lora_rank={args.lora_rank}",
            f"--learning_rate={args.lr}",
            f"--emb_learning_rate={args.emb_lr}",
            f"--train_batch_size={args.batch_size//num_gpu}",
            f"--max_train_steps={args.total_steps}",
            f"--checkpointing_steps={args.total_steps // 5}",
            "--gradient_accumulation_steps=1",
            f"--unet_params_to_train={args.train_params}",
            f"--augment={args.augment}",
            f"--augment_p={args.augment_p}",
            f"--null_prob={args.null_prob}",
            f"--kpl_weight={args.kpl_weight}",
            "--template=imagenet_small",
            "--mixed_precision=fp16",
        ]
        if args.num_samples is not None:
            cmd.append(f"--num_samples={args.num_samples}")
        if not args.no_inversion:
            cmd.append("--augment_inversion")
        if args.no_weighted_sample:
            cmd.append("--disable_weighted_sample")
        if args.augment == "none":
            cmd.append("--center_crop")
        if args.mixing:
            cmd.append("--mixing")
        subprocess.run(torchrun_cmd + cmd)

        # save cmd as text file
        cmd_txt = "\n".join(cmd)
        with open(f"{outdir}/{name}/cmd.txt", "w") as file:
            file.write(cmd_txt)
        shutil.copy("train_textboost.py", f"{outdir}/{name}/train_textboost.py")


if __name__ == "__main__":
    args = parse_args()
    main(args)
