import glob
import os

import textboost.text_encoders.clip as clip
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.transforms import v2

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
    "a {0} in the jungle",
    "a {0} in the snow",
    "a {0} on the beach",
    "a {0} on a cobblestone street",
    "a {0} on top of pink fabric",
    "a {0} on top of a wooden floor",
    "a {0} with a city in the background",
    "a {0} with a mountain in the background",
    "a {0} with a blue house in the background",
    "a {0} on top of a purple rug in a forest",
    "a {0} with a wheat field in the background",
    "a {0} with a tree and autumn leaves in the background",
    "a {0} with the Eiffel Tower in the background",
    "a {0} floating on top of water",
    "a {0} floating in an ocean of milk",
    "a {0} on top of green grass with sunflowers around it",
    "a {0} on top of a mirror",
    "a {0} on top of the sidewalk in a crowded street",
    "a {0} on top of a dirt road",
    "a {0} on top of a white rug",
    "a red {0}",
    "a purple {0}",
    "a shiny {0}",
    "a wet {0}",
    "a cube shaped {0}",
]

LIVE_PROMPTS = [
    "a {0} in the jungle",
    "a {0} in the snow",
    "a {0} on the beach",
    "a {0} on a cobblestone street",
    "a {0} on top of pink fabric",
    "a {0} on top of a wooden floor",
    "a {0} with a city in the background",
    "a {0} with a mountain in the background",
    "a {0} with a blue house in the background",
    "a {0} on top of a purple rug in a forest",
    "a {0} wearing a red hat",
    "a {0} wearing a santa hat",
    "a {0} wearing a rainbow scarf",
    "a {0} wearing a black top hat and a monocle",
    "a {0} in a chef outfit",
    "a {0} in a firefighter outfit",
    "a {0} in a police outfit",
    "a {0} wearing pink glasses",
    "a {0} wearing a yellow shirt",
    "a {0} in a purple wizard outfit",
    "a red {0}",
    "a purple {0}",
    "a shiny {0}",
    "a wet {0}",
    "a cube shaped {0}",
]


class InstanceDataset(torch.utils.data.Dataset):
    def __init__(self, root, instance, transform=None):
        self.root = root
        self.transform = transform
        # self.files = glob.glob(f"{root}/{instance}/*/*.png")  # root/instance/seed/prompt.png
        self.files = glob.glob(
            f"{root}/*/{instance}/*.png"
        )  # root/seed/instance/prompt.png

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        basename = os.path.basename(path)
        instance = os.path.dirname(path).split("/")[-1]
        prompt = basename.replace(".png", "").replace("_", " ")

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, instance, prompt


def is_live(instance):
    cls = INSTANCES[instance]
    return cls in ("dog", "cat")


def clip_image_score(sample_dir, data_dict, args, device):
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    # model, preprocess = clip.load("ViT-B/32", device=device)  # same as Custom Diffusion.
    model.eval().requires_grad_(False)
    preprocess = v2.Compose(
        [
            v2.Resize((512, 512)),  # TODO: 1024 for SDXL.
            preprocess,
        ]
    )

    scores = []
    for instance in data_dict.keys():
        train_data = []
        root = data_dict[instance]["path"]
        images = os.listdir(root)
        for image in images:
            img = Image.open(os.path.join(root, image)).convert("RGB")

            if args.mask_path is not None:
                mask_file = os.path.splitext(image)[0] + ".png"
                mask = os.path.join(args.mask_path, instance, mask_file)
                mask = Image.open(mask).convert("L")
                img = np.asarray(img) * np.asarray(mask)[:, :, None]
                img = Image.fromarray(img)

            train_data.append(preprocess(img))
        train_images = torch.stack(train_data)
        train_feats = model.encode_image(train_images.to(device))
        train_feats = train_feats.float().cpu().numpy()

        dataset = InstanceDataset(sample_dir, instance, transform=preprocess)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        print(instance, len(dataset))  # should be 3000.

        for images, _, _ in dataloader:
            image_features = model.encode_image(images.to(device)).float()
            image_features = image_features.cpu().numpy()

            sim_matrix = cosine_similarity(image_features, train_feats)
            sim_matrix[sim_matrix < 0] = 0.0
            # print(sim_matrix.shape)
            scores.append(sim_matrix.reshape(-1))
    scores = np.concatenate(scores)

    print(f"Total samples: {len(scores)}")
    print(f"CLIP-I: {scores.mean():.3f} +/- {scores.std():.3f}")
    return scores
