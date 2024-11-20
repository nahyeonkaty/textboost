#!/usr/bin/env python
import argparse
import csv
import glob
import os
from pprint import pprint

import clip
import ImageReward as RM
import numpy as np
import t2v_metrics
import torch
from diffusers import (DiffusionPipeline, DPMSolverMultistepScheduler,
                       UNet2DConditionModel)
from diffusers.loaders import LoraLoaderMixin
from diffusers.training_utils import _set_state_dict_into_text_encoder
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import CLIPTextModel

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="path to model")
parser.add_argument(
    "--token-format",
    type=str,
    default="<INSTANCE> SUBJECT",
    help=(
        "Token format for the prompt ",
        "[sks SUBJECT] for DreamBooth models, ",
        "[<INSTANCE>] for Textual Inversion models, "
        "[<INSTANCE> SUBJECT] for CustomDiffusion and TextBoost."
    )
)
parser.add_argument("--outdir", type=str, default="./benchmarks")
parser.add_argument("--checkpoint", type=int, default=None)
parser.add_argument("--instances", type=str, nargs="+", default=None)
parser.add_argument("--skip-gen", action="store_true")
parser.add_argument("--metric", type=str, nargs="+", default=["clip-t", "clip-i"])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--dreambooth-path", type=str, default="./data/dreambooth")
parser.add_argument("--train-dir", type=str, default="./data/dreambooth_n1_train")
parser.add_argument("--val-dir", type=str, default="./data/dreambooth_n1_val")
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--output-desc", type=str, default=None)

STABLE_DIFFUSION = {
    "sd14": "CompVis/stable-diffusion-v1-4",
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd21base": "stabilityai/stable-diffusion-2-1-base",
    "sd21": "stabilityai/stable-diffusion-2-1",
}

INSTANCES = {
    "backpack": "backpack",
    "backpack_dog": "backpack",
    "bear_plushie": "stuffed animal",
    "berry_bowl": "bowl",
    "can": "can",
    "candle": "candle",
    "cat": "cat",
    "cat2": "cat",
    "clock": "clock",
    "colorful_sneaker": "sneaker",
    "dog": "dog",
    "dog2": "dog",
    "dog3": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
    "duck_toy": "toy",
    "fancy_boot": "boot",
    "grey_sloth_plushie": "stuffed animal",
    "monster_toy": "toy",
    "pink_sunglasses": "glasses",
    "poop_emoji": "toy",
    "rc_car": "toy",
    "red_cartoon": "cartoon",
    "robot_toy": "toy",
    "shiny_sneaker": "sneaker",
    "teapot": "teapot",
    "vase": "vase",
    "wolf_plushie": "stuffed animal",
}

OBJ_PROMPTS = [
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on a cobblestone street',
    'a {0} on top of pink fabric',
    'a {0} on top of a wooden floor',
    'a {0} with a city in the background',
    'a {0} with a mountain in the background',
    'a {0} with a blue house in the background',
    'a {0} on top of a purple rug in a forest',
    'a {0} with a wheat field in the background',
    'a {0} with a tree and autumn leaves in the background',
    'a {0} with the Eiffel Tower in the background',
    'a {0} floating on top of water',
    'a {0} floating in an ocean of milk',
    'a {0} on top of green grass with sunflowers around it',
    'a {0} on top of a mirror',
    'a {0} on top of the sidewalk in a crowded street',
    'a {0} on top of a dirt road',
    'a {0} on top of a white rug',
    'a red {0}',
    'a purple {0}',
    'a shiny {0}',
    'a wet {0}',
    'a cube shaped {0}'
]

LIVE_PROMPTS = [
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on a cobblestone street',
    'a {0} on top of pink fabric',
    'a {0} on top of a wooden floor',
    'a {0} with a city in the background',
    'a {0} with a mountain in the background',
    'a {0} with a blue house in the background',
    'a {0} on top of a purple rug in a forest',
    'a {0} wearing a red hat',
    'a {0} wearing a santa hat',
    'a {0} wearing a rainbow scarf',
    'a {0} wearing a black top hat and a monocle',
    'a {0} in a chef outfit',
    'a {0} in a firefighter outfit',
    'a {0} in a police outfit',
    'a {0} wearing pink glasses',
    'a {0} wearing a yellow shirt',
    'a {0} in a purple wizard outfit',
    'a red {0}',
    'a purple {0}',
    'a shiny {0}',
    'a wet {0}',
    'a cube shaped {0}'
]


def is_live(instance):
    cls = INSTANCES[instance]
    if cls in ("cat", "dog"):
        return True
    return False


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, return_str=False):
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


def load_dreambooth_pipeline(pipeline, model_path):
    subfolders = os.listdir(model_path)
    if "unet" in subfolders:
        pipeline.unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet",
        )
        print("Loaded UNet weights")
    if "text_encoder" in subfolders:
        pipeline.text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder",
        )
    return pipeline


def load_lora_pipeline(pipeline, model_path):
    pipeline.load_lora_weights(model_path)
    # lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(model_path)
    # unet_state_dict = {
    #     f'{k.replace("unet.", "")}': v
    #     for k, v in lora_state_dict.items()
    # }
    # for k in unet_state_dict:
    #     if "lora_A" in k:
    #         lora_rank = unet_state_dict[k].shape[0]
    #         lora_config = LoraConfig(
    #             r=lora_rank,
    #             lora_alpha=lora_rank,
    #             init_lora_weights="gaussian",
    #             target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    #         )
    #         pipeline.unet.add_adapter(lora_config)
    #         break
    # unet_state_dict = convert_state_dict_to_diffusers(unet_state_dict, original_type=torch.float32)
    # incompatible_keys = set_peft_model_state_dict(pipeline.unet, unet_state_dict, adapter_name="default")

    # if incompatible_keys is not None:
    #     # check only for unexpected keys
    #     unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    #     if unexpected_keys:
    #         print(f"Unexpected keys in the state dict: {unexpected_keys}")
    print("Loaded U-Net LoRA weights from checkpoint")
    return pipeline


def load_customdiffusion_pipeline(pipeline, model_path):
    weight_name = "pytorch_custom_diffusion_weights.bin"
    custom_diffusion_state_dict = torch.load(
        os.path.join(model_path, weight_name),
        map_location="cpu",
    )
    attn_state_dict = dict()
    for key, value in custom_diffusion_state_dict.items():
        _key = key.replace(".processor", "")
        _key = _key.replace("_custom_diffusion", "")
        attn_state_dict[_key] = value
    _, unexpected = pipeline.unet.load_state_dict(attn_state_dict, strict=False)
    assert len(unexpected) == 0, f"Unexpected keys in the state dict: {unexpected}"
    try:
        instance = model_path.split("/")[-2]
        print(model_path)
        print(instance)
        pipeline.load_textual_inversion(
            os.path.join(model_path, f"{instance}.bin"),
        )
    except:
        pipeline.load_textual_inversion(
            os.path.join(model_path, "<new>.bin"),
        )
    return pipeline


def load_textboost_pipeline(pipeline, model_path):
    try:
        text_encoder_path = os.path.join(model_path, "text_encoder")
        pipeline.text_encoder.load_adapter(text_encoder_path)
        print("Loaded text encoder LoRA weights")
    except:
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
        STABLE_DIFFUSION[pretrained_model],
        use_safetensors=True,
        safety_checker=None,
    )

    if model_path is None:
        pass
    elif "db" in model_path:
        pipeline = load_dreambooth_pipeline(pipeline, model_path)
    elif "lora" in model_path:
        pipeline = load_lora_pipeline(pipeline, model_path)
    elif "ti" in model_path:
        if "checkpoint" in model_path:
            step = os.path.basename(model_path).split("-")[-1]
            model_path = os.path.dirname(model_path)
            embedding = os.path.join(model_path, f"learned_embeds-steps-{step}.safetensors")
        else:
            embedding = os.path.join(model_path, "learned_embeds.safetensors")
        pipeline.load_textual_inversion(embedding)
    elif "cd" in model_path:
        pipeline = load_customdiffusion_pipeline(pipeline, model_path)
    elif "auginv" in model_path:
        if "checkpoint" in model_path:
            step = os.path.basename(model_path).split("-")[-1]
            model_path = os.path.dirname(model_path)
        embedding = os.path.join(model_path, "learned_embeds0.bin")
        pipeline.load_textual_inversion(embedding)
    elif "tb" in model_path:
        pipeline = load_textboost_pipeline(pipeline, model_path)
    else:
        raise ValueError(f"Unknown model path: {model_path}")

    pipeline.vae.eval().requires_grad_(False)
    pipeline.unet.eval().requires_grad_(False)
    pipeline.text_encoder.eval().requires_grad_(False)

    pipeline = pipeline.to(dtype=dtype)

    pipeline.set_progress_bar_config(disable=True)
    torch.cuda.empty_cache()
    return pipeline


def generate_from_pipeline(
        pipeline,
        instance,
        size,
        identifier,
        token_format,
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

    latent = torch.randn(1, 4, size, size, dtype=pipeline.dtype, device=device)

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
            latents=latent.repeat(len(prompts), 1, 1, 1),
        ).images

        for prompt, image in zip(prompts, images):
            dst = os.path.join(outdir, f"seed{seed}", instance)
            os.makedirs(dst, exist_ok=True)
            filename = f"{prompt.replace(identifier, cls).replace(' ', '_')}.png"
            image.save(
                os.path.join(dst, filename)
            )
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
        subdirs = list(filter(lambda x: os.path.isdir(os.path.join(args.path, x)), subdirs))
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
        if num_vectors is not None:
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
                token_format=args.token_format,
                seed=seed,
                outdir=outdir,
                batch_size=16,
                device=device,
            )
    return outdir


def clip_t(generated_image_path, device, w=2.5):
    score = t2v_metrics.CLIPScore(model='openai:ViT-L-14-336', device=device)
    score.eval().requires_grad_(False)

    def _path_to_prompt(path):
        basename = os.path.basename(path)  # prompt.png
        return basename.replace(".png", "").replace("_", " ")

    image_paths = glob.glob(f"{generated_image_path}/*/*.png")  # root/instance/prompt.png
    dataset = [
        {"images": [image_path], "texts": [_path_to_prompt(image_path)]}
        for image_path in image_paths
    ]
    scores = score.batch_forward(dataset=dataset, batch_size=32)
    # shape: len(dataset), len(dataset[0]["images"], len(dataset[0]["texts"])
    scores = w * scores  # scale mo

    del score
    torch.cuda.empty_cache()
    print(f"Total samples: {scores.size(0)}")
    print(f"CLIP-T: {scores.mean():.3f} +/- {scores.std():.3f}")
    return {"clip_score": scores}


def clip_i(args, generated_image_path, device):
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    # model, preprocess = clip.load("ViT-B/32", device=device)  # same as Custom Diffusion.
    model.eval().requires_grad_(False)
    preprocess = v2.Compose([
        v2.Resize((512, 512)),
        preprocess,
    ])

    N = int(args.path.split("-")[2].split("n")[-1])
    train_dir = args.train_dir
    test_dir = args.val_dir

    seen_images = sorted(glob.glob(os.path.join(train_dir, "*/*.*")))
    unseen_images = sorted(glob.glob(os.path.join(test_dir, "*/*.*")))
    instance_to_id = {}

    id = 0
    seen_data = {}
    for image in seen_images:
        instance = os.path.basename(os.path.dirname(image))
        image = Image.open(image).convert("RGB")
        if instance in seen_data:
            seen_data[instance].append(preprocess(image))
        else:
            seen_data[instance] = [preprocess(image)]
            instance_to_id[instance] = id
            id += 1

    unseen_data = {}
    for image in unseen_images:
        instance = os.path.basename(os.path.dirname(image))
        image = Image.open(image).convert("RGB")
        if instance in unseen_data:
            unseen_data[instance].append(preprocess(image))
        else:
            unseen_data[instance] = [preprocess(image)]

    seen_scores = []
    unseen_scores = []
    n = 0
    for instance in os.listdir(generated_image_path):
        images = sorted(glob.glob(os.path.join(generated_image_path, instance, "*.png")))
        images = torch.stack([
            preprocess(Image.open(image).convert("RGB"))
            for image in images
        ])
        image_features = model.encode_image(images.to(device))  # 25 prompts per instance

        # Compare to seen images.
        train_batch = torch.stack(seen_data[instance])
        seen_feature = model.encode_image(train_batch.to(device))  # num_seen, D
        for seen_feat in seen_feature.unbind(0):
            seen_feat = seen_feat.unsqueeze(0)
            seen_score = torch.cosine_similarity(image_features, seen_feat, dim=1)
            seen_score = torch.maximum(seen_score, torch.zeros_like(seen_score))
            seen_scores.append(seen_score)

        # Compare to unseen images.
        test_batch = torch.stack(unseen_data[instance])
        unseen_feature = model.encode_image(test_batch.to(device))  # num_seen, D
        for unseen_feat in unseen_feature.unbind(0):
            unseen_feat = unseen_feat.unsqueeze(0)
            unseen_score = torch.cosine_similarity(image_features, unseen_feat, dim=1)
            unseen_score = torch.maximum(unseen_score, torch.zeros_like(unseen_score))
            unseen_scores.append(unseen_score)

        n += images.shape[0]

    seen_scores = torch.cat(seen_scores)
    unseen_scores = torch.cat(unseen_scores)

    print(f"Total samples: {n}")
    print(f"CLIP-I (seen)  : {seen_scores.mean():.3f} +/- {seen_scores.std():.3f}")
    print(f"CLIP-I (unseen): {unseen_scores.mean():.3f} +/- {unseen_scores.std():.3f}")
    return {"clip_i_seen": seen_scores, "clip_i_unseen": unseen_scores}


def dino_score(args, generated_image_path, device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval().requires_grad_(False).to(device)
    preprocess = v2.Compose([
        v2.Resize((512, 512)),
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dreambooth = {}
    dreambooth_images = sorted(glob.glob(os.path.join(args.dreambooth_path, "**/*.*")))
    instance_to_id = {}

    id = 0
    for image in dreambooth_images:
        instance = os.path.basename(os.path.dirname(image))
        image = Image.open(image).convert("RGB")
        if instance in dreambooth:
            dreambooth[instance].append(preprocess(image))
        else:
            dreambooth[instance] = [preprocess(image)]
            instance_to_id[instance] = id
            id += 1

    max_samples = 0
    num_samples = {}
    for instance, images in dreambooth.items():
        id = instance_to_id[instance]
        max_samples = max(max_samples, len(images))
        num_samples[id] = len(images)

    db_batch = torch.zeros(len(dreambooth), max_samples, 3, 224, 224)
    # db_batch = torch.zeros(len(dreambooth), max_samples, 3, 336, 336)
    for instance, images in dreambooth.items():
        id = instance_to_id[instance]
        db_batch[id, :num_samples[id]] = torch.stack(images)


    N = int(args.path.split("-")[2].split("n")[-1])
    dataloader = torch.utils.data.DataLoader(
        Dataset(generated_image_path, transform=preprocess),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    seen_scores = []
    unseen_scores = []
    n = 0
    for image, instance, prompt in dataloader:
        instance = list(map(lambda x: instance_to_id[x], instance))
        instance = torch.as_tensor(instance, dtype=torch.long)
        token = clip.tokenize(prompt)
        image_feature = model(image.to(device))

        # Compare to seen images.
        train_batch = db_batch[instance][:, :N, :, :, :].to(device)
        seen_i_score = 0.0
        for i, train_image in enumerate(train_batch.unbind(1)):
            seen_i_score += torch.cosine_similarity(image_feature, model(train_image), dim=-1)
        seen_i_score /= N

        # Compare to unseen images.
        unseen_batch = db_batch[instance][:, N:, :, :, :].to(device)
        unseen_i_score = 0.0
        num_unseen = torch.as_tensor([num_samples[id.item()]-N for id in instance], device=device)
        for i, unseen_image in enumerate(unseen_batch.unbind(1)):
            sim = torch.cosine_similarity(image_feature, model(unseen_image), dim=-1)
            mask = torch.ones(len(instance), device=device) * i
            mask = mask < torch.as_tensor(num_unseen, device=device)
            sim = mask * sim
            unseen_i_score += sim
        unseen_i_score /= num_unseen

        n += image.shape[0]
        seen_scores.append(seen_i_score)
        unseen_scores.append(unseen_i_score)

    seen_scores = torch.cat(seen_scores)
    unseen_scores = torch.cat(unseen_scores)

    print(f"Total samples: {n}")
    print(f"DINO: {seen_scores.mean():.3f} +/- {seen_scores.std():.3f}")
    print(f"DINO (unseen): {unseen_scores.mean():.3f} +/- {seen_scores.std():.3f}")
    return {"dino": seen_scores, "dino_unseen": unseen_scores}


def radio_score(args, generated_image_path, device):
    # model_version = "radio_v2.5-l"  # ViT-L/16
    # model_version = "radio_v2.5-b"  # ViT-B/16
    model_version = "radio_v2.1"  # ViT-H/16-CPE
    model = torch.hub.load(
        "NVlabs/RADIO", 'radio_model', version=model_version,
        progress=True, skip_validation=True,
    ).eval().requires_grad_(False).to(device)

    dreambooth = {}
    dreambooth_images = sorted(glob.glob(os.path.join(args.dreambooth_path, "**/*.*")))
    instance_to_id = {}

    preprocess = v2.Compose([
        v2.Resize((512, 512)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    id = 0
    for image in dreambooth_images:
        instance = os.path.basename(os.path.dirname(image))
        image = Image.open(image).convert("RGB")
        if instance in dreambooth:
            dreambooth[instance].append(preprocess(image))
        else:
            dreambooth[instance] = [preprocess(image)]
            instance_to_id[instance] = id
            id += 1

    max_samples = 0
    num_samples = {}
    for instance, images in dreambooth.items():
        id = instance_to_id[instance]
        max_samples = max(max_samples, len(images))
        num_samples[id] = len(images)

    db_batch = torch.zeros(len(dreambooth), max_samples, 3, 512, 512)
    for instance, images in dreambooth.items():
        id = instance_to_id[instance]
        db_batch[id, :num_samples[id]] = torch.stack(images)


    N = int(args.path.split("-")[2].split("n")[-1])
    dataloader = torch.utils.data.DataLoader(
        Dataset(generated_image_path, transform=preprocess),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    seen_scores = []
    unseen_scores = []
    n = 0
    for image, instance, prompt in dataloader:
        instance = list(map(lambda x: instance_to_id[x], instance))
        instance = torch.as_tensor(instance, dtype=torch.long)
        summary, spatial_features = model(image.to(device))

        # Compare to seen images.
        train_batch = db_batch[instance][:, :N, :, :, :].to(device)
        seen_score = 0.0
        for i, train_image in enumerate(train_batch.unbind(1)):
            summary2, _ = model(train_image)
            seen_score += torch.cosine_similarity(summary, summary2, dim=-1)
        seen_score /= N

        # Compare to unseen images.
        unseen_batch = db_batch[instance][:, N:, :, :, :].to(device)
        unseen_score = 0.0
        num_unseen = torch.as_tensor([num_samples[id.item()]-N for id in instance], device=device)
        for i, unseen_image in enumerate(unseen_batch.unbind(1)):
            summary2, _ = model(unseen_image)
            sim = torch.cosine_similarity(summary, summary2, dim=-1)
            mask = torch.ones(len(instance), device=device) * i
            mask = mask < torch.as_tensor(num_unseen, device=device)
            sim = mask * sim
            unseen_score += sim
        unseen_score /= num_unseen

        n += image.shape[0]
        seen_scores.append(seen_score)
        unseen_scores.append(unseen_score)

    seen_scores = torch.cat(seen_scores)
    unseen_scores = torch.cat(unseen_scores)

    print(f"Total samples: {n}")
    print(f"RADIO-seen: {seen_scores.mean():.3f} +/- {seen_scores.std():.3f}")
    print(f"RADIO-unseen: {unseen_scores.mean():.3f} +/- {seen_scores.std():.3f}")
    return {"radio_seen": seen_scores, "radio_unseen": unseen_scores}


def vqa_score(args, generated_image_path, device):
    import t2v_metrics
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl', device=device)
    clip_flant5_score.eval().requires_grad_(False)

    dataset = []
    files = glob.glob(f"{generated_image_path}/*/*/*.png")
    for file in files:
        text = os.path.basename(file).replace(".png", "").replace("_", " ")
        dataset.append({
            "images": [file], "texts": [text]
        })
    print("Number of samples:", len(dataset))

    scores = clip_flant5_score.batch_forward(dataset=dataset, batch_size=32)

    del clip_flant5_score
    torch.cuda.empty_cache()
    return {"vqa_score": scores}


def image_reward(args, generated_image_path, device="cuda"):
    model = RM.load("ImageReward-v1.0")
    model.eval().requires_grad_(False)

    image_list = sorted(glob.glob(f"{generated_image_path}/*/*/*.png"))

    image_rewards = []
    for image in tqdm(image_list):
        prompt = os.path.basename(image).replace(".png", "").replace("_", " ")
        score = model.score(prompt, image)
        image_rewards.append(score)

    image_rewards = np.asarray(image_rewards)

    print(f"Total samples: {len(image_list)}")
    print(f"Image reward: {image_rewards.mean():.3f} +/- {image_rewards.std():.3f}")
    return {"image_reward": image_rewards}


@torch.inference_mode()
def main(args):
    if args.path.endswith("/"):
        args.path = args.path[:-1]

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    generated_image_path = generate(args, device)

    # Save scores to file.
    metric = "_".join(args.metric)
    seeds = ",".join(map(str, args.seeds))
    ckpt = f"_ckpt{args.checkpoint}" if args.checkpoint is not None else "_last"
    desc = f"_{args.output_desc}" if args.output_desc is not None else ""
    filename = f"{metric}{ckpt}{desc}.csv"
    # If not exists, create the file and write header.
    with open(os.path.join(args.path, filename), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "clip-t", "clip-i-seen", "clip-i-unseen"])

    score_dict = {}
    for seed in args.seeds:
        path_with_seed = os.path.join(generated_image_path, f"seed{seed}")

        score = {}
        if "clip-t" in args.metric:
            score.update(clip_t(path_with_seed, device))
        if "clip-i" in args.metric:
            score.update(clip_i(args, path_with_seed, device))

        if "vqa" in args.metric:
            score.update(vqa_score(args, path_with_seed, device))
        if "dino" in args.metric:
            score.update(dino_score(args, path_with_seed, device))
        if "image_reward" in args.metric:
            score.update(image_reward(args, path_with_seed, device))
        if "radio" in args.metric:
            score.update(radio_score(args, path_with_seed, device))

        score_dict[f"seed{seed}"] = score
        # Save scores to file.
        with open(os.path.join(args.path, filename), "a") as f:
            writer = csv.writer(f)
            line = [
                str(seed),
                score["clip_score"].mean().item(),
                score["clip_i_seen"].mean().item(),
                score["clip_i_unseen"].mean().item(),
            ]
            writer.writerow(line)
    # pprint(score_dict, sort_dicts=False)

    filename = f"{metric}{ckpt}-{seeds}.txt"
    with open(os.path.join(args.path, filename), "a+") as f:
        for seed_key, seed_score in score_dict.items():
            f.write(f"{seed_key}\n")
            for key, value in seed_score.items():
                f.write(f"{key}: ")
                f.write(f"{value.mean():.3f} +/- {value.std():.3f}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
