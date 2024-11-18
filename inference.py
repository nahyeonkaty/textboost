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
    "sd1.4": "CompVis/stable-diffusion-v1-4",
    # "sd1.5": "runwayml/stable-diffusion-v1-5",
    "sd1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd2.1": "stabilityai/stable-diffusion-2-1",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to model")
    parser.add_argument("--model", type=str, default="sd1.5")
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


def load_textboost_pipeline(pipeline, model_path):
    if "checkpoint" in model_path:
        lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(model_path)
        text_encoder_state_dict = {
            f'{k.replace("text_encoder.", "")}': v
            for k, v in lora_state_dict.items()
        }
        for k in text_encoder_state_dict:
            if "lora_A" in k:
                lora_rank = text_encoder_state_dict[k].shape[0]
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank,
                    init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                )
                pipeline.text_encoder.add_adapter(lora_config)
                break
        text_encoder_state_dict = convert_state_dict_to_diffusers(text_encoder_state_dict)
        incompatible_keys = _set_state_dict_into_text_encoder(
            lora_state_dict, prefix="text_encoder.", text_encoder=pipeline.text_encoder
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                print(f"Unexpected keys in the state dict: {unexpected_keys}")
        print("Loaded text encoder LoRA weights from checkpoint")
    else:
        text_encoder_path = os.path.join(model_path, "text_encoder")
        if "adapter_config.json" in os.listdir(text_encoder_path):
            pipeline.text_encoder.load_adapter(text_encoder_path)
            print("Loaded text encoder LoRA weights")
        else:
            pipeline.text_encoder = CLIPTextModel.from_pretrained(
                model_path, subfolder="text_encoder",
            )
            print("Loaded text encoder weights")

    # Load learned embeddings
    embeddings = list(filter(lambda x: x.endswith(".bin"), os.listdir(model_path)))
    # remove the learned embeddings from the list of files
    for embedding in sorted(embeddings):
        if os.path.basename(embedding) in (
            "optimizer.bin", "scheduler.bin",
        ):
            continue
        emb_path = os.path.join(model_path, embedding)
        pipeline.load_textual_inversion(emb_path)
        print(f"Loaded learned embeddings from {emb_path}")
    return pipeline


def load_pipeline(model_path, pretrained_model, dtype=torch.float16):
    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model,
        use_safetensors=True,
        safety_checker=None,
    )
    pipeline = load_textboost_pipeline(pipeline, model_path)

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
