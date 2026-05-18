#!/usr/bin/env python
import argparse
import json
import os

import clip
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/dreambooth_n1.json")
    return parser.parse_args()


def main():
    args = parse_arguments()

    with open(args.data, "r") as f:
        data = json.load(f)

    device = "cuda:1"
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

    image_scores = []
    for name, metadata in data.items():
        train_image = metadata["path"]
        root = os.path.dirname(train_image)
        images = [os.path.join(root, img) for img in os.listdir(root)]
        train_image = metadata["path"]
        images.remove(train_image)
        print(train_image)
        print(images)
        train_image = preprocess(Image.open(train_image)).unsqueeze(0).to(device)
        images = torch.stack([preprocess(Image.open(img)).to(device) for img in images])

        feat = model.encode_image(train_image).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feats = model.encode_image(images).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)

        image_score = F.cosine_similarity(feat, feats).cpu().numpy()
        print(image_score)
        image_scores.append(image_score.mean())
    image_scores = np.asarray(image_scores).mean()
    print(image_scores)


if __name__ == "__main__":
    main()
