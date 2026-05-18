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

from textboost.evaluation.dreambooth import (
    INSTANCES,
    LIVE_PROMPTS,
    OBJ_PROMPTS,
    is_live,
)
from textboost.pipelines.loaders import load_sd_pipeline
from textboost.pipelines.sana import TextBoostSanaPipeline
from textboost.t2v_compat import import_t2v_metrics

t2v_metrics, _ = import_t2v_metrics()


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
    parser.add_argument(
        "--train_data", type=str, default="./datasets/dreambooth_n1.json"
    )
    parser.add_argument("--dreambooth_path", type=str, default="./datasets/dreambooth")
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="./images")
    parser.add_argument("--output_desc", type=str, default=None)
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for generation."
    )

    parser.add_argument("--metric", type=str, nargs="+", default=["vqa", "dino"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--skip_gen", action="store_true")

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--instances", type=str, nargs="+", default=None)
    return parser.parse_args()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, return_str: bool = False):
        self.root = root
        self.transform = transform
        self.return_str = return_str
        self.files = glob.glob(f"{root}/*/*.png")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        basename = os.path.basename(path)
        instance = os.path.dirname(path).split("/")[-1]
        prompt = basename.replace(".png", "").replace("_", " ")

        if self.return_str:
            return path, instance, prompt

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, instance, prompt


def load_pipeline(
    model_path: str,
    pretrained_model: str,
    key: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[TextBoostSanaPipeline | load_sd_pipeline, list[str]]:
    print(pretrained_model)
    if "sana" in pretrained_model:
        if "sana1.5" in pretrained_model:
            v = 1.5
        else:
            v = 1.0
        pipeline, identifiers = TextBoostSanaPipeline.from_checkpoint(
            model=SANA_MODEL_DICT[f"v{v}"][key],
            checkpoint_path=model_path,
        )
        pipeline.vae.to(dtype=torch.float32)
        pipeline.transformer.to(dtype=torch.bfloat16)
        pipeline.text_encoder.to(dtype=torch.bfloat16)
    else:
        pipeline, identifiers = load_sd_pipeline(
            model=SD_MODEL_DICT[pretrained_model],
            checkpoint_path=model_path,
        )
        pipeline = pipeline.to(dtype=dtype)

    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.eval().requires_grad_(False)
    pipeline.text_encoder.eval().requires_grad_(False)

    torch.cuda.empty_cache()
    return pipeline, identifiers


def generate_from_pipeline(
    pipeline,
    instance,
    z_shape,
    identifier,
    seed,
    outdir,
    batch_size: int = 16,
    num_inference_steps: int = 25,
    device: str | torch.device = "cuda",
) -> None:
    assert instance in INSTANCES, f"Invalid instance: {instance}"
    prompt_list = LIVE_PROMPTS if is_live(instance) else OBJ_PROMPTS

    if outdir.endswith("/"):
        outdir = outdir[:-1]

    cls = INSTANCES[instance]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    print(f"[seed {seed} - Identifiers: {identifier}")

    latent = torch.empty(1, *z_shape, device=device).normal_(generator=generator)

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
            # prompt_embeds=prompt_embeds,
            num_inference_steps=num_inference_steps,
            # guidance_scale=7.5,  # NOTE: default value.
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
    if args.output_desc is not None:
        outdir = outdir + f"_{args.output_desc}"
    if args.skip_gen:
        return outdir

    key = None
    num_inference_steps = 25
    if args.model is not None:
        model = args.model
        z_shape = [4, 64, 64]
    elif "sd14" in args.path:
        model = "sd14"
        z_shape = [4, 64, 64]
    elif "sd15" in args.path:
        model = "sd15"
        z_shape = [4, 64, 64]
    elif "sd21base" in args.path:
        model = "sd21base"
        z_shape = [4, 64, 64]
    elif "sd21" in args.path:
        model = "sd21"
        z_shape = [4, 96, 96]
    elif "sana" in args.path:
        splits = args.path.split("-")
        model = splits[1]
        key = splits[2]
        z_shape = [32, 32, 32]
        num_inference_steps = 20
    else:
        raise NotImplementedError

    for instance in tqdm(instances):
        ckpt = f"checkpoint-{args.checkpoint}" if args.checkpoint is not None else ""
        model_path = os.path.join(args.path, instance, ckpt)

        pipeline, identifiers = load_pipeline(
            model_path,
            model,
            key,
            dtype=torch.float16,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline = pipeline.to(device)

        identifier = args.token_format.replace("<INSTANCE>", identifiers[0])
        # files = os.listdir(model_path)
        # num_vectors = len(list(filter(lambda x: x.startswith(instance), files)))
        # identifier = args.token_format.replace("INSTANCE", instance)
        # if num_vectors > 1:
        #     tokens = []
        #     for i in range(num_vectors):
        #         tokens.append(identifier.replace(">", f"_{i}>"))
        #     identifier = " ".join(tokens)
        identifier = identifier.replace("SUBJECT", INSTANCES[instance])

        for seed in args.seeds:
            generate_from_pipeline(
                pipeline=pipeline,
                instance=instance,
                z_shape=z_shape,
                identifier=identifier,
                seed=seed,
                outdir=outdir,
                batch_size=args.batch_size,
                num_inference_steps=num_inference_steps,
                device=device,
            )
    return outdir


def clip_score(generated_image_path, device):
    score = t2v_metrics.CLIPScore(model="openai:ViT-L-14-336", device=device)
    score.eval().requires_grad_(False)

    def _path_to_prompt(path):
        basename = os.path.basename(path)  # prompt.png
        return basename.replace(".png", "").replace("_", " ")

    # root/instance/prompt.png
    image_paths = glob.glob(f"{generated_image_path}/*/*.png")
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
    return {"clip": scores}


def clip_image_score(generated_image_path, args, device):
    try:
        import clip
    except ModuleNotFoundError as e:
        raise ImportError(
            "`clip` package is required for `clip_i` metric. "
            "Install OpenAI CLIP to enable this metric."
        ) from e
    if getattr(clip, "__textboost_stub__", False):
        raise ImportError(
            "`clip` package is required for `clip_i` metric. "
            "Install OpenAI CLIP to enable this metric."
        )

    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    # model, preprocess = clip.load("ViT-B/32", device=device)  # same as Custom Diffusion.
    model.eval().requires_grad_(False)
    preprocess = v2.Compose(
        [
            v2.Resize((512, 512)),
            preprocess,
        ]
    )

    data_dir = "datasets/dreambooth"  # TODO: use arguments.
    with open(args.train_data, "r") as file:
        train_data = json.load(file)

    seen_data = {}
    unseen_data = {}
    for instance, metadata in train_data.items():
        image_path = metadata["path"]
        image = Image.open(image_path).convert("RGB")
        if args.mask_dir is not None:
            fname = os.path.basename(image_path).split(".")[0]
            mask_path = os.path.join(args.mask_dir, instance, f"{fname}.png")
            mask = Image.open(mask_path).convert("RGB")
            image = np.asarray(image)
            mask = np.asarray(mask) / 255
            image = image * mask
            image = Image.fromarray(image.astype(np.uint8))
        seen_data[instance] = [preprocess(image)]

        filename = os.path.basename(image_path)
        images = os.listdir(os.path.join(data_dir, instance))
        unseen_data[instance] = []
        for image in sorted(images):
            if image != filename:
                image_path = os.path.join(data_dir, instance, image)
                unseen_image = Image.open(image_path).convert("RGB")
                if args.mask_dir is not None:
                    fname = os.path.basename(image_path).split(".")[0]
                    mask_path = os.path.join(args.mask_dir, instance, f"{fname}.png")
                    mask = Image.open(mask_path).convert("RGB")
                    unseen_image = np.asarray(unseen_image)
                    mask = np.asarray(mask) / 255
                    unseen_image = unseen_image * mask
                    unseen_image = Image.fromarray(unseen_image.astype(np.uint8))
                unseen_data[instance].append(preprocess(unseen_image))

    seen_scores = []
    unseen_scores = []
    for instance in os.listdir(generated_image_path):
        images = sorted(
            glob.glob(os.path.join(generated_image_path, instance, "*.png"))
        )
        images = torch.stack(
            [preprocess(Image.open(image).convert("RGB")) for image in images]
        )
        image_features = model.encode_image(
            images.to(device)
        ).float()  # 25 prompts per instance

        # Compare to seen images.
        train_batch = torch.stack(seen_data[instance])
        seen_feature = model.encode_image(train_batch.to(device)).float()  # num_seen, D
        for seen_feat in seen_feature.unbind(0):
            seen_feat = seen_feat.unsqueeze(0)
            seen_score = torch.cosine_similarity(image_features, seen_feat, dim=1)
            seen_score[seen_score < 0.0] = 0.0
            seen_scores.append(seen_score)

        # Compare to unseen images.
        test_batch = torch.stack(unseen_data[instance])
        unseen_feature = model.encode_image(
            test_batch.to(device)
        ).float()  # num_seen, D
        for unseen_feat in unseen_feature.unbind(0):
            unseen_feat = unseen_feat.unsqueeze(0)
            unseen_score = torch.cosine_similarity(image_features, unseen_feat, dim=1)
            unseen_score[unseen_score < 0.0] = 0.0
            unseen_scores.append(unseen_score)

    seen_scores = torch.cat(seen_scores)
    unseen_scores = torch.cat(unseen_scores)

    print(f"Total samples: {seen_scores.size(0) + unseen_scores.size(0)}")
    print(f"CLIP-I (seen)  : {seen_scores.mean():.3f} +/- {seen_scores.std():.3f}")
    print(f"CLIP-I (unseen): {unseen_scores.mean():.3f} +/- {unseen_scores.std():.3f}")
    return {"clip_i": seen_scores, "clip_i_unseen": unseen_scores}


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

    data_dir = "datasets/dreambooth"  # TODO: use arguments.
    with open(args.train_data, "r") as file:
        train_data = json.load(file)

    seen_data = {}
    unseen_data = {}
    for instance, metadata in train_data.items():
        image_path = metadata["path"]
        image = Image.open(image_path).convert("RGB")
        if args.mask_dir is not None:
            fname = os.path.basename(image_path).split(".")[0]
            mask_path = os.path.join(args.mask_dir, instance, f"{fname}.png")
            mask = Image.open(mask_path).convert("RGB")
            image = np.asarray(image)
            mask = np.asarray(mask) / 255
            image = image * mask
            image = Image.fromarray(image.astype(np.uint8))
        seen_data[instance] = [preprocess(image)]

        filename = os.path.basename(image_path)
        images = os.listdir(os.path.join(data_dir, instance))
        unseen_data[instance] = []
        for image in sorted(images):
            if image != filename:
                image_path = os.path.join(data_dir, instance, image)
                unseen_image = Image.open(image_path).convert("RGB")
                if args.mask_dir is not None:
                    fname = os.path.basename(image_path).split(".")[0]
                    mask_path = os.path.join(args.mask_dir, instance, f"{fname}.png")
                    mask = Image.open(mask_path).convert("RGB")
                    unseen_image = np.asarray(unseen_image)
                    mask = np.asarray(mask) / 255
                    unseen_image = unseen_image * mask
                    unseen_image = Image.fromarray(unseen_image.astype(np.uint8))
                unseen_data[instance].append(preprocess(unseen_image))

    seen_scores = []
    unseen_scores = []
    for instance in os.listdir(generated_image_path):
        images = sorted(
            glob.glob(os.path.join(generated_image_path, instance, "*.png"))
        )
        images = torch.stack(
            [preprocess(Image.open(image).convert("RGB")) for image in images]
        )
        image_features = model(images.to(device)).float()  # 25 prompts per instance

        # Compare to seen images.
        train_batch = torch.stack(seen_data[instance])
        seen_feature = model(train_batch.to(device)).float()  # num_seen, D
        for seen_feat in seen_feature.unbind(0):
            seen_feat = seen_feat.unsqueeze(0)
            seen_score = torch.cosine_similarity(image_features, seen_feat, dim=1)
            seen_score[seen_score < 0.0] = 0.0
            seen_scores.append(seen_score)

        # Compare to unseen images.
        test_batch = torch.stack(unseen_data[instance])
        unseen_feature = model(test_batch.to(device)).float()  # num_seen, D
        for unseen_feat in unseen_feature.unbind(0):
            unseen_feat = unseen_feat.unsqueeze(0)
            unseen_score = torch.cosine_similarity(image_features, unseen_feat, dim=1)
            unseen_score[unseen_score < 0.0] = 0.0
            unseen_scores.append(unseen_score)

    seen_scores = torch.cat(seen_scores)
    unseen_scores = torch.cat(unseen_scores)

    print(f"Total samples: {seen_scores.size(0) + unseen_scores.size(0)}")
    print(f"DINOv2 (seen)  : {seen_scores.mean():.3f} +/- {seen_scores.std():.3f}")
    print(f"DINOv2 (unseen): {unseen_scores.mean():.3f} +/- {unseen_scores.std():.3f}")
    return {"dino": seen_scores, "dino_unseen": unseen_scores}


def vqa_score(generated_image_path, device):
    clip_flant5_score = t2v_metrics.VQAScore(model="clip-flant5-xxl", device=device)
    clip_flant5_score.eval().requires_grad_(False)

    dataset = []
    # root/seed/instance/prompt.png
    files = glob.glob(f"{generated_image_path}/*/*.png")
    for file in files:
        text = os.path.basename(file).replace(".png", "").replace("_", " ")
        dataset.append({"images": [file], "texts": [text]})
    print("Number of samples:", len(dataset))
    scores = clip_flant5_score.batch_forward(dataset=dataset, batch_size=32)

    del clip_flant5_score
    torch.cuda.empty_cache()

    print(f"VQA score: {scores.mean():.3f} +/- {scores.std():.3f}")
    return {"vqa": scores}


@torch.inference_mode()
def main():
    args = parse_arguments()

    if args.path.endswith("/"):
        args.path = args.path[:-1]

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    generated_image_path = generate(args, device)

    score_dict = {
        seed: {
            # Image-text scores
            "clip": torch.tensor([0.0]),
            "vqa": torch.tensor([0.0]),
            # Image-image scores
            "clip_i": torch.tensor([0.0]),
            "clip_i_unseen": torch.tensor([0.0]),
            "dino": torch.tensor([0.0]),
            "dino_unseen": torch.tensor([0.0]),
        }
        for seed in args.seeds
    }

    for seed in args.seeds:
        path_with_seed = os.path.join(generated_image_path, f"seed{seed}")

        if "clip" in args.metric:
            score_dict[seed].update(clip_score(path_with_seed, device))
        if "vqa" in args.metric:
            score_dict[seed].update(vqa_score(path_with_seed, device))

        if "clip_i" in args.metric:
            score_dict[seed].update(clip_image_score(path_with_seed, args, device))
        if "dino" in args.metric:
            score_dict[seed].update(dino_score(path_with_seed, args, device))

    # Save scores to file.
    desc = f"_{args.output_desc}" if args.output_desc is not None else ""
    filename = f"metric{desc}.csv"
    headline = ["checkpoint", "seed"] + list(score_dict[args.seeds[0]].keys())
    print(headline)
    result_file = os.path.join(args.path, filename)
    if not os.path.exists(result_file):
        with open(result_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(headline)
    with open(result_file, "a") as f:
        writer = csv.writer(f)
        for seed, score in score_dict.items():
            line = [str(args.checkpoint), str(seed)]
            line += list(map(lambda x: str(x.mean().cpu().item()), score.values()))
            print(line)
            writer.writerow(line)


if __name__ == "__main__":
    main()
