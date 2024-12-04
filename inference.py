#!/usr/bin/env python3
import argparse
import os

import torch
from diffusers import (DiffusionPipeline, DPMSolverMultistepScheduler)
from diffusers.loaders import LoraLoaderMixin
from diffusers.training_utils import _set_state_dict_into_text_encoder
from diffusers.utils import convert_state_dict_to_diffusers, make_image_grid
from peft import LoraConfig
from PIL import Image
from transformers import CLIPTextModel


STABLE_DIFFUSION = {
    "sd14": "CompVis/stable-diffusion-v1-4",
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd21base": "stabilityai/stable-diffusion-2-1-base",
    "sd21": "stabilityai/stable-diffusion-2-1",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to model")
    parser.add_argument("--model", type=str, default="sd21base")
    parser.add_argument(
        "--prompt",
        type=str,
        default="photo of a <dog> dog",
        help=(
            "[sks SUBJECT] for DreamBooth models, ",
            "[<INSTANCE>] for Textual Inversion models, "
            "[<INSTANCE> SUBJECT] for CustomDiffusion and TextBoost."
        )
    )
    parser.add_argument("--outdir", type=str, default="./benchmarks")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if args.model in STABLE_DIFFUSION.keys():
        args.model = STABLE_DIFFUSION[args.model]
    return args


def load_pipeline(model_path, pretrained_model, dtype=torch.float16):
    pipeline = DiffusionPipeline.from_pretrained(
        STABLE_DIFFUSION[pretrained_model],
        use_safetensors=True,
        safety_checker=None,
    )

    # Load TextBoost pipeline.
    text_encoder_path = os.path.join(model_path, "text_encoder")
    pipeline.text_encoder.load_adapter(text_encoder_path, "default")
    print("Loaded text encoder LoRA weights")
    pipeline.text_encoder.set_adapter("default")

    # Load learned embeddings
    embeddings = list(filter(lambda x: x.endswith(".bin"), os.listdir(model_path)))
    # remove the learned embeddings from the list of files
    for embedding in sorted(embeddings):
        if os.path.basename(embedding) in ("optimizer.bin", "scheduler.bin"):
            continue
        emb_path = os.path.join(model_path, embedding)
        pipeline.load_textual_inversion(emb_path)
        print(f"Loaded learned embeddings from {emb_path}")

    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.eval().requires_grad_(False)
    pipeline.unet.eval().requires_grad_(False)
    pipeline.text_encoder.eval().requires_grad_(False)
    pipeline = pipeline.to(dtype=dtype)

    torch.cuda.empty_cache()
    return pipeline


@torch.inference_mode()
def main(args):
    if args.path.endswith("/"):
        args.path = args.path[:-1]

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    pipeline = load_pipeline(args.path, args.model)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(device)

    generator = [
        torch.Generator(device).manual_seed(seed) for seed in args.seeds
    ]
    images = pipeline(
        prompt=args.prompt,
        num_images_per_prompt=len(generator),
        generator=generator,
        return_images=True,
        device=device,
    ).images


    if args.output is not None:
        output = args.output
        image = make_image_grid(images, 1, len(args.seeds))
        image.save(output)
    else:
        for seed, image in zip(args.seeds, images):
            output = args.prompt.replace(" ", "_") + f"_{seed}.jpg"
            image.save(output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
