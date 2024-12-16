#!/usr/bin/env python3
import argparse
import csv
import glob
import os

import clip
import t2v_metrics
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from textboost.text_encoder import TextBoostModel


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

def parse_args():
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
    parser.add_argument("--metric", type=str, nargs="+", default=["clip-t", "clip-i", "vqa"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dreambooth-path", type=str, default="./data/dreambooth")
    parser.add_argument("--train-dir", type=str, default="./data/dreambooth_n1_train")
    parser.add_argument("--val-dir", type=str, default="./data/dreambooth_n1_val")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output-desc", type=str, default=None)
    return parser.parse_args()


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


def load_pipeline(model_path, pretrained_model, dtype=torch.float16):
    text_encoder = TextBoostModel.from_pretrained(
        STABLE_DIFFUSION[pretrained_model], subfolder="text_encoder",
    )
    start_embedding = torch.load(
        f"assets/start_emb_{pretrained_model}.pt", map_location="cpu"
    )
    text_encoder.set_null_embedding(start_embedding)
    print(text_encoder.null_embedding)

    pipeline = DiffusionPipeline.from_pretrained(
        STABLE_DIFFUSION[pretrained_model],
        text_encoder=text_encoder,
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
    score = t2v_metrics.CLIPScore(model='openai:ViT-L-14-336', device=device)
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
    scores = score.batch_forward(dataset=dataset, batch_size=32)
    # shape: len(dataset), len(dataset[0]["images"], len(dataset[0]["texts"])

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
    return {"clip_i": seen_scores, "clip_i_unseen": unseen_scores}


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


def vqa_score(args, generated_image_path, device):
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl', device=device)
    clip_flant5_score.eval().requires_grad_(False)

    dataset = []
    # root/instance/prompt.png
    files = glob.glob(f"{generated_image_path}/*/*.png")
    for file in files:
        text = os.path.basename(file).replace(".png", "").replace("_", " ")
        dataset.append({
            "images": [file], "texts": [text]
        })
    print("Number of samples:", len(dataset))
    scores = clip_flant5_score.batch_forward(dataset=dataset, batch_size=32)

    del clip_flant5_score
    torch.cuda.empty_cache()

    print(f"VQA score: {scores.mean():.3f} +/- {scores.std():.3f}")
    return {"vqa_score": scores}


@torch.inference_mode()
def main(args):
    if args.path.endswith("/"):
        args.path = args.path[:-1]

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    generated_image_path = generate(args, device)

    # Save scores to file.
    ckpt = f"_ckpt{args.checkpoint}" if args.checkpoint is not None else "_last"
    desc = f"_{args.output_desc}" if args.output_desc is not None else ""
    filename = f"metric{ckpt}{desc}.csv"


    score_dict = {
        seed: {
            # Image-text scores
            "clip_score": torch.tensor([0.0]),
            "vqa_score": torch.tensor([0.0]),
            # Image-image scores
            "clip_i": torch.tensor([0.0]),
            "clip_i_unseen": torch.tensor([0.0]),
            "dino": torch.tensor([0.0]),
            "dino_unseen": torch.tensor([0.0]),
        }
        for seed in args.seeds
    }

    # If not exists, create the file and write header.
    with open(os.path.join(args.path, filename), "w") as f:
        writer = csv.writer(f)
        headline = ["seed"] + list(score_dict[args.seeds[0]].keys())
        writer.writerow(headline)

    for seed in args.seeds:
        path_with_seed = os.path.join(generated_image_path, f"seed{seed}")

        if "clip-t" in args.metric:
            score_dict[seed].update(clip_score(path_with_seed, device))
        if "vqa" in args.metric:
            score_dict[seed].update(vqa_score(args, path_with_seed, device))

        if "clip-i" in args.metric:
            score_dict[seed].update(clip_i(args, path_with_seed, device))
        if "dino" in args.metric:
            score_dict[seed].update(dino_score(args, path_with_seed, device))

    # Save scores to file.
    print(headline)
    with open(os.path.join(args.path, filename), "a") as f:
        writer = csv.writer(f)
        for seed, score in score_dict.items():
            line = (
                [str(seed)]
                + list(
                    map(lambda x: f"{x.mean().cpu().item():.3f}", score.values())
                )
            )
            print(line)
            writer.writerow(line)


if __name__ == "__main__":
    args = parse_args()
    main(args)
