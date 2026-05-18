#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm
from sklearn.metrics import pairwise

from textboost.t2v_compat import import_t2v_metrics
from textboost.evaluation.dreambooth import (
    INSTANCES,
    LIVE_PROMPTS,
    OBJ_PROMPTS,
    InstanceDataset,
    clip_image_score,
    is_live,
)
from textboost.pipelines.loaders import load_sd_pipeline

t2v_metrics, _ = import_t2v_metrics()


STABLE_DIFFUSION = {
    "sd14": "CompVis/stable-diffusion-v1-4",
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd21base": "stabilityai/stable-diffusion-2-1-base",
    "sd21": "stabilityai/stable-diffusion-2-1",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to model")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument(
        "--token_format",
        type=str,
        default="<INSTANCE> SUBJECT",
        help=(
            "Token format for the prompt ",
            "[sks SUBJECT] for DreamBooth models, ",
            "[<INSTANCE>] for Textual Inversion models, "
            "[<INSTANCE> SUBJECT] for CustomDiffusion and TextBoost.",
        ),
    )
    parser.add_argument("--train_data", type=str, default="./datasets/dreambooth.json")
    parser.add_argument("--dreambooth_dir", type=str, default="./datasets/dreambooth")
    parser.add_argument("--mask_dir", type=str, default="datasets/dreambooth_mask")

    parser.add_argument("--instances", type=str, nargs="+", default=None)
    parser.add_argument("--skip_gen", action="store_true")
    parser.add_argument("--metric", type=str, nargs="+", default=["vqa", "dino"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="./images")
    parser.add_argument("--desc", type=str, default=None)
    return parser.parse_args()


def load_pipeline(model_path, pretrained_model, dtype=torch.float16):
    pipeline, identifiers = load_sd_pipeline(
        model=STABLE_DIFFUSION[pretrained_model],
        checkpoint_path=model_path,
    )

    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.eval().requires_grad_(False)
    pipeline.unet.eval().requires_grad_(False)
    pipeline.text_encoder.eval().requires_grad_(False)
    pipeline = pipeline.to(dtype=dtype)

    torch.cuda.empty_cache()
    return pipeline


def generate_from_pipeline(
    pipeline,
    instance,
    size,
    identifier,
    seed,
    outdir,
    batch_size=8,
    device="cuda",
):
    assert instance in INSTANCES, f"Invalid instance: {instance}"
    prompt_list = LIVE_PROMPTS if is_live(instance) else OBJ_PROMPTS

    if outdir.endswith("/"):
        outdir = outdir[:-1]

    cls = INSTANCES[instance]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    print(f"[seed {seed} - Identifiers: {identifier}")

    latent = torch.empty(1, 4, size, size, device=device).normal_(generator=generator)

    i = 0
    while i < len(prompt_list):
        prompts = []
        for _ in range(batch_size):
            prompts.append(prompt_list[i].format(identifier))
            i += 1
            if i >= len(prompt_list):
                break

        print(len(prompts))
        print(prompts)

        images = pipeline(
            prompt=prompts,
            num_inference_steps=25,
            guidance_scale=7.5,  # NOTE: default value.
            latents=latent.repeat(len(prompts), 1, 1, 1).to(pipeline.dtype),
        ).images

        for prompt, image in zip(prompts, images):
            dst = os.path.join(outdir, f"seed{seed}", instance)
            os.makedirs(dst, exist_ok=True)
            filename = f"{prompt.replace(identifier, cls).replace(' ', '_')}.png"
            image.save(os.path.join(dst, filename))
    del pipeline, generator


def generate(args, device):
    if args.instances is not None:
        instances = {}
        for name, cls in INSTANCES.items():
            if name in args.instances:
                instances[name] = cls
    else:
        instances = INSTANCES

        subdirs = os.listdir(args.path)
        subdirs = list(
            filter(lambda x: os.path.isdir(os.path.join(args.path, x)), subdirs)
        )
        # for instance in INSTANCES.keys():
        #     assert instance in subdirs, f"Missing instance: {instance}"
        # assert len(subdirs) == 30, f"Invalid number of instances: {len(subdirs)}"

    if args.outdir.endswith("/"):
        args.outdir = args.outdir[:-1]
    if args.path.endswith("/"):
        args.path = args.path[:-1]

    if args.checkpoint is not None:
        basename = f"{os.path.basename(args.path)}-{args.checkpoint}"
    else:
        basename = os.path.basename(args.path)
    outdir = os.path.join(args.outdir, basename)
    if args.desc is not None:
        outdir = outdir + f"_{args.desc}"
    if args.skip_gen:
        return outdir

    if args.model is not None:
        model = args.model
        size = 64
    elif "sd14" in args.path:
        model = "sd14"
        size = 64
    elif "sd15" in args.path:
        model = "sd15"
        size = 64
    elif "sd21base" in args.path:
        model = "sd21base"
        size = 64
    elif "sd21" in args.path:
        model = "sd21"
        size = 96

    for instance in tqdm(instances):
        ckpt = f"checkpoint-{args.checkpoint}" if args.checkpoint is not None else ""
        model_path = os.path.join(args.path, instance, ckpt)

        pipeline = load_pipeline(model_path, model, dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline = pipeline.to(device)
        print(pipeline.tokenizer)

        # identifier = identifier.format(INSTANCES[instance])
        files = os.listdir(model_path)
        num_vectors = len(list(filter(lambda x: x.startswith(instance), files)))
        identifier = args.token_format.replace("INSTANCE", instance)
        if num_vectors > 1:
            tokens = []
            for i in range(num_vectors):
                tokens.append(identifier.replace(">", f"_{i}>"))
            identifier = " ".join(tokens)
        identifier = identifier.replace("SUBJECT", INSTANCES[instance])

        for seed in args.seeds:
            generate_from_pipeline(
                pipeline=pipeline,
                instance=instance,
                size=size,
                identifier=identifier,
                seed=seed,
                outdir=outdir,
                batch_size=16,
                device=device,
            )
    return outdir


def clip_score(generated_image_path, device):
    score = t2v_metrics.CLIPScore(model="openai:ViT-L-14-336", device=device)
    score.eval().requires_grad_(False)

    def _path_to_prompt(path):
        basename = os.path.basename(path)  # prompt.png
        return basename.replace(".png", "").replace("_", " ")

    # root/seed/instance/prompt.png
    image_paths = glob.glob(f"{generated_image_path}/*/*/*.png")
    dataset = [
        {"images": [image_path], "texts": [_path_to_prompt(image_path)]}
        for image_path in image_paths
    ]
    scores = score.batch_forward(dataset=dataset, batch_size=64)
    # shape: len(dataset), len(dataset[0]["images"], len(dataset[0]["texts"])

    del score
    torch.cuda.empty_cache()
    print(f"Total samples: {scores.size(0)}")
    print(f"CLIP-T: {scores.mean():.3f} +/- {scores.std():.3f}")
    return scores.cpu().numpy()


def dino_score(generated_image_path, args, device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval().requires_grad_(False).to(device)
    preprocess = v2.Compose(
        [
            v2.Resize((512, 512)),
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    with open(args.train_data, "r") as file:
        data = json.load(file)

    scores = []
    for instance in data.keys():
        images = []
        for f in os.listdir(os.path.join(args.dreambooth_dir, instance)):
            image = Image.open(os.path.join(args.dreambooth_dir, instance, f))
            image = np.asarray(image)
            maskf = f.replace(".jpg", ".png")
            mask = Image.open(os.path.join(args.mask_dir, instance, maskf))
            mask = mask.convert("RGB")
            mask = np.asarray(mask) / 255

            image = image * mask
            image = Image.fromarray(image.astype(np.uint8))
            images.append(preprocess(image))
        images = torch.stack(images)
        train_feats = model(images.to(device)).float().cpu().numpy()

        dataset = InstanceDataset(generated_image_path, instance, transform=preprocess)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        for image, instances, prompt in dataloader:
            # print(instances)
            image_feats = model(image.to(device)).float().cpu().numpy()  # B, D

            sim_matrix = pairwise.cosine_similarity(image_feats, train_feats)
            sim_matrix[sim_matrix < 0] = 0.0  # B, N
            # print(sim_matrix.shape)
            score = sim_matrix.reshape(-1)
            # print(score.shape)
            scores.append(score)

    scores = np.concat(scores)

    print(f"Total samples: {scores.shape[0]}")
    print(f"DINO: {scores.mean():.3f} +/- {scores.std():.3f}")
    return scores


def vqa_score(generated_image_path, device):
    clip_flant5_score = t2v_metrics.VQAScore(model="clip-flant5-xxl", device=device)
    clip_flant5_score.eval().requires_grad_(False)

    dataset = []
    # root/seed/instance/prompt.png
    files = glob.glob(f"{generated_image_path}/*/*/*.png")
    for file in files:
        text = os.path.basename(file).replace(".png", "").replace("_", " ")
        dataset.append({"images": [file], "texts": [text]})
    print("Number of samples:", len(dataset))
    scores = clip_flant5_score.batch_forward(dataset=dataset, batch_size=32)

    del clip_flant5_score
    torch.cuda.empty_cache()

    print(f"VQA score: {scores.mean():.3f} +/- {scores.std():.3f}")
    return scores.cpu().numpy()


@torch.inference_mode()
def main():
    args = parse_arguments()

    if args.path.endswith("/"):
        args.path = args.path[:-1]

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    generated_image_path = generate(args, device)
    with open(args.train_data, "r") as file:
        data = json.load(file)

    score_dict = {
        # Image-text scores
        "clip": np.zeros(1),
        "vqa": np.zeros(1),
        # Image-image scores
        "clip_i": np.zeros(1),
        "dino": np.zeros(1),
    }

    with open(args.train_data, "r") as file:
        data = json.load(file)

    if "clip-i" in args.metric:
        result = clip_image_score(generated_image_path, data, args, device)
        score_dict["clip_i"] = result
    if "dino" in args.metric:
        score_dict["dino"] = dino_score(generated_image_path, args, device)

    if "clip" in args.metric:
        score_dict["clip"] = clip_score(generated_image_path, device)

    if "vqa" in args.metric:
        score_dict["vqa"] = vqa_score(generated_image_path, device)

    # Save scores to file.
    # If not exists, create the file and write header.
    csvfile = os.path.join(args.path, "metric.csv")
    headline = ["checkpoint"] + list(score_dict.keys())
    if not os.path.exists(csvfile):
        with open(csvfile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(headline)
    print(headline)
    with open(csvfile, "a") as f:
        writer = csv.writer(f)
        line = [args.checkpoint] + list(
            map(lambda x: str(x.mean()), score_dict.values())
        )
        writer.writerow(line)
        print(line)


if __name__ == "__main__":
    main()
