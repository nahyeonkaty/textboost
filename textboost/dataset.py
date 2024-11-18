import json
import os
import random
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision.transforms import v2

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

textboost_templates = [
    # "{}",
    # "a {}",
    # "one {}"
    # "the {}",
    # "photo of a {}",
    "{} with background",
    "a {} with background",
    # "one {} with background"
    "the {} with background",
    "photo of a {} with background",
]


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def get_images_path(data_root, max_samples=None):
    data_root = Path(data_root)
    if not data_root.exists():
        raise ValueError("Data root doesn't exists.")
    images_path = list(data_root.iterdir())
    images_path.sort()
    if max_samples is not None:
        return images_path[:max_samples]
    print("Number of samples: ", len(images_path))
    return images_path


def get_images_path_recursive(data_root, max_samples=None):
    data_root = Path(data_root)
    if not data_root.exists():
        raise ValueError("Data root doesn't exists.")
    images_path = list(data_root.rglob("*"))
    images_path = [p for p in images_path if p.is_file()]
    images_path.sort()
    if max_samples is not None:
        return images_path[:max_samples]
    print("Number of samples: ", len(images_path))
    return images_path


class DrawBench(torch.utils.data.Dataset):
    url = "https://raw.githubusercontent.com/google-research/google-research/master/dpok/dataset/drawbench/data_meta.json"

    def __init__(self, tokenizer, num_samples=None):
        import requests
        text = requests.get(self.url).text
        self.data = []
        for i, line in enumerate(text.split('\n')[1:-1]):
            if i % 3 == 0:
                prompt = line.split('"')[1]
            elif i % 3 == 1:
                category = line.split('"')[3].lower()
            else:
                self.data.append((prompt, category))
                del prompt, category

        if num_samples is not None:
            self.data = self.data[:num_samples]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt, cls = self.data[index]
        text_inputs = tokenize_prompt(self.tokenizer, prompt)
        return {
            "prompt": prompt,
            "class": cls,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
        }

    def __repr__(self):
        lines = [f"{self.__class__.__name__}(num_samples={len(self)})"]
        for prompt, category in self.data:
            lines.append(f"  {prompt} ({category})")
        return '\n'.join(lines)


class InstructPix2PixDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, json_file, num_samples=None):
        with open(json_file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
            # print(data)
        self.data = []
        for i, line in enumerate(data):
            self.data.append(line["input"])
            self.data.append(line["output"])

        if num_samples is not None:
            self.data = self.data[:num_samples]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = self.data[index]
        text_inputs = tokenize_prompt(self.tokenizer, prompt)
        return {
            "prompt": prompt,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
        }

    def __repr__(self):
        lines = [f"{self.__class__.__name__}(num_samples={len(self)})"]
        for prompt in self.data:
            lines.append(f"  {prompt}")
        return '\n'.join(lines)


class PriorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source,
        tokenizer,
        additional_template=None,
        additional_category=None,
        template_prob=0.1,
        null_prob=0.1,
    ):
        super().__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.template_prob = template_prob
        self.null_prob = null_prob

        for prompt in source.data:
            self.data.append(prompt)

        try:
            template = {
                "imagenet_small": imagenet_templates_small,
                "imagenet_style_small": imagenet_style_templates_small,
                "tb": textboost_templates,
                "tepa": textboost_templates,
            }[additional_template]
        except:
            template = [additional_template]
        if not isinstance(additional_category, list):
            additional_category = [additional_category]

        self.template_data = []
        for prompt in template:
            for c in additional_category:
                p = prompt.format(c)
                self.template_data.append(p)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rand = random.random()
        if rand < self.null_prob:
            prompt = ""
        elif rand < self.null_prob + self.template_prob:
            prompt = random.choice(self.template_data)
        else:
            prompt = self.data[index]
        text_inputs = tokenize_prompt(self.tokenizer, prompt)
        return {
            "prompt": prompt,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
        }

    def __repr__(self):
        lines = [f"{self.__class__.__name__}(num_samples={len(self)})"]
        lines.append(f"null prob: {self.null_prob:.2f}")
        for prompt, category in self.data:
            lines.append(f"  {prompt}")
        return '\n'.join(lines)

    @staticmethod
    def collate_fn(samples):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [example["attention_mask"] for example in samples]

        input_ids = torch.cat(input_ids, dim=0)

        batch = {
            "prompt": [sample["prompt"] for sample in samples],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return batch


class TextBoostDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        concepts_list,
        tokenizer,
        num_instance=None,
        template="a {}",
        prior_data_root=None,
        class_token=None,
        num_prior=None,
        size=512,
        center_crop=False,
        augment_pipe=None,
        augment_prior: bool = False,
    ):
        try:
            self.template = {
                "imagenet_small": imagenet_templates_small,
                "imagenet_style_small": imagenet_style_templates_small,
                "textboost": textboost_templates,
                "tepa": textboost_templates,
            }[template]
        except:
            self.template = [template]
        print(self.template)

        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_path = []
        for concept in concepts_list:
            images_path = [
                (x, concept["instance_token"])
                for x in get_images_path(concept["instance_data_dir"], num_instance)
            ]
            self.instance_images_path.extend(images_path)

        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        self.class_token = class_token
        if prior_data_root is not None:
            self.prior_data_root = Path(prior_data_root)
            self.prior_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.prior_data_root.iterdir())
            if num_prior is not None:
                self.num_prior_images = min(len(self.class_images_path), num_prior)
            else:
                self.num_prior_images = len(self.class_images_path)
            self._length = max(self.num_prior_images, self.num_instance_images)
        else:
            self.prior_data_root = None

        self.image_transforms = v2.Compose([
            v2.Resize(size, interpolation=v2.InterpolationMode.LANCZOS),
            v2.CenterCrop(size) if center_crop else v2.RandomCrop(size),
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.augment_pipe = augment_pipe
        self.augment_prior = augment_prior

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        sample = {}

        instance_image, instance_token = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        prompt_idx = random.randint(0, len(self.template) - 1)
        prompt = self.template[prompt_idx].format(instance_token)
        if self.augment_pipe is not None:
            instance_image, prompt, mask = self.augment_pipe(instance_image, prompt)
            if mask is not None:
                mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
                sample["mask"] = mask

        text_inputs = tokenize_prompt(self.tokenizer, prompt)
        sample["instance_images"] = self.image_transforms(instance_image)
        sample["instance_prompt_ids"] = text_inputs.input_ids
        sample["instance_attention_mask"] = text_inputs.attention_mask

        if self.prior_data_root:
            prior_path = self.class_images_path[index % self.num_prior_images]
            prior_image = Image.open(prior_path)
            prior_image = exif_transpose(prior_image)
            prior_image = prior_image.convert("RGB")

            if self.class_token is not None:
                prompt = self.template[prompt_idx].format(self.class_token)
            else:
                prompt = (
                    os.path.basename(prior_path)
                    .split("-")[1]
                    .split(".")[0]
                    .replace("_", " ")
                )

            if self.augment_prior and self.augment_pipe is not None:
                prior_image, prompt, mask = self.augment_pipe(prior_image, prompt)
                if mask is not None:
                    mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
                    sample["prior_mask"] = mask
            if "mask" in sample and "prior_mask" not in sample:
                sample["prior_mask"] = torch.ones_like(sample["mask"])

            sample["class_images"] = self.image_transforms(prior_image)
            prior_text_inputs = tokenize_prompt(self.tokenizer, prompt)
            sample["class_prompt_ids"] = prior_text_inputs.input_ids
            sample["class_attention_mask"] = prior_text_inputs.attention_mask
        return sample

    @staticmethod
    def collate_fn(samples, with_prior_preservation=False):
        has_attention_mask = "instance_attention_mask" in samples[0]

        input_ids = [sample["instance_prompt_ids"] for sample in samples]
        pixel_values = [sample["instance_images"] for sample in samples]

        if has_attention_mask:
            attention_mask = [example["instance_attention_mask"] for example in samples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in samples]
            pixel_values += [example["class_images"] for example in samples]
            if has_attention_mask:
                attention_mask += [example["class_attention_mask"] for example in samples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if "mask" in samples[0]:
            mask = [sample["mask"] for sample in samples]
            if "prior_mask" in samples[0]:
                mask += [sample["prior_mask"] for sample in samples]
            batch["mask"] = torch.stack(mask)

        if has_attention_mask:
            batch["attention_mask"] = attention_mask

        return batch


class JsonDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file,
        instance,
        instance_token,
        tokenizer,
        template="a {}",
        prior_data_root=None,
        class_token=None,
        num_prior=None,
        size=512,
        center_crop=False,
        max_samples=999999,
        augment_pipe=None,
    ):
        assert os.path.exists(json_file)
        if isinstance(instance, str):
            instance = [instance]
        if isinstance(instance_token, str):
            instance_token = [instance_token]

        try:
            self.template = {
                "imagenet_small": imagenet_templates_small,
                "imagenet_style_small": imagenet_style_templates_small,
                "tepa": textboost_templates,
            }[template]
        except:
            self.template = [template]
        print(self.template)

        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        with open(json_file, "r") as f:
            data = json.load(f)

        self.instance_token = {}
        self.data = []
        for inst, token in zip(instance, instance_token):
            self.instance_token[inst] = token
            for i in range(max_samples):
                _data = data[inst][str(i)]
                _data["instance_token"] = token
                self.data.append(_data)

        self.class_token = class_token
        if prior_data_root is not None:
            self.prior_data_root = Path(prior_data_root)
            self.prior_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.prior_data_root.iterdir())
            if num_prior is not None:
                self.num_prior_images = min(len(self.class_images_path), num_prior)
            else:
                self.num_prior_images = len(self.class_images_path)
            self._length = max(self.num_prior_images, self.num_instance_images)
        else:
            self.prior_data_root = None

        self.image_transforms = v2.Compose([
            v2.Resize(size),
            v2.CenterCrop(size) if center_crop else v2.RandomCrop(size),
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.augment_pipe = augment_pipe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {}

        data = self.data[index]

        cache_path = os.path.join(
            ".cache/",
            data["url"].replace("https://", "").replace("/", "_").replace("?raw=true", ""),
        )
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            image = Image.open(requests.get(data["url"], stream=True).raw)
            image.save(cache_path)
        else:
            image = Image.open(cache_path)

        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        token = data["instance_token"]
        prompt = data["template"].format(token)
        if self.augment_pipe is not None:
            image, prompt, mask = self.augment_pipe(image, prompt)
            if mask is not None:
                mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
                sample["mask"] = mask

        text_inputs = tokenize_prompt(self.tokenizer, prompt)
        sample["instance_images"] = self.image_transforms(image)
        sample["instance_prompt_ids"] = text_inputs.input_ids
        sample["instance_attention_mask"] = text_inputs.attention_mask
        return sample

    @staticmethod
    def collate_fn(samples, with_prior_preservation=False):
        has_attention_mask = "instance_attention_mask" in samples[0]

        input_ids = [sample["instance_prompt_ids"] for sample in samples]
        pixel_values = [sample["instance_images"] for sample in samples]

        if has_attention_mask:
            attention_mask = [example["instance_attention_mask"] for example in samples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in samples]
            pixel_values += [example["class_images"] for example in samples]
            if has_attention_mask:
                attention_mask += [example["class_attention_mask"] for example in samples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if "mask" in samples[0]:
            mask = [sample["mask"] for sample in samples]
            if "prior_mask" in samples[0]:
                mask += [sample["prior_mask"] for sample in samples]
            batch["mask"] = torch.stack(mask)

        if has_attention_mask:
            batch["attention_mask"] = attention_mask

        return batch


class FolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        center_crop=False,
        augment_pipe=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.images_path = get_images_path_recursive(data_root)

        self.image_transforms = v2.Compose([
            v2.Resize(size),
            v2.CenterCrop(size) if center_crop else v2.RandomCrop(size),
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.augment_pipe = augment_pipe

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        sample = {}

        images_path = self.images_path[index]
        image = Image.open(images_path)
        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        filename = os.path.basename(images_path)
        filename = filename.split(".")[0]
        prompt = filename.split("-")[-1].replace("_", " ")
        if self.augment_pipe is not None:
            image, prompt, mask = self.augment_pipe(image, prompt)
            if mask is not None:
                mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
                sample["mask"] = mask

        text_inputs = tokenize_prompt(self.tokenizer, prompt)
        sample["images"] = self.image_transforms(image)
        sample["prompt_ids"] = text_inputs.input_ids
        sample["attention_mask"] = text_inputs.attention_mask
        return sample

    @staticmethod
    def collate_fn(samples):
        has_attention_mask = "attention_mask" in samples[0]

        input_ids = [sample["prompt_ids"] for sample in samples]
        pixel_values = [sample["images"] for sample in samples]

        if has_attention_mask:
            attention_mask = [example["attention_mask"] for example in samples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if "mask" in samples[0]:
            mask = [sample["mask"] for sample in samples]
            if "prior_mask" in samples[0]:
                mask += [sample["prior_mask"] for sample in samples]
            batch["mask"] = torch.stack(mask)

        if has_attention_mask:
            batch["attention_mask"] = attention_mask

        return batch


class PromptDataset(torch.utils.data.Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(
            self,
            token,
            template="a photo of {}",
            num_samples=None,
        ):
        if isinstance(token, str):
            token = [token]
        self.token = token
        if isinstance(template, str):
            self.template = [template]
        elif template == "imagenet_small":
            self.template = imagenet_templates_small
        elif template == "imagenet_style_small":
            self.template = imagenet_style_templates_small
        else:
            raise ValueError("Invalid template.")
        self.num_samples = num_samples

    def __len__(self):
        if self.num_samples is None:
            return len(self.template)
        return self.num_samples

    def __getitem__(self, index):
        sample = {}
        token = random.choice(self.token)
        sample["prompt"] = random.choice(self.template).format(token)
        sample["index"] = index
        return sample


class StyleDrop(torch.utils.data.Dataset):
    def __init__(self):
        # Image attributes
        self.urls = [
            "https://images.unsplash.com/photo-1578926078693-4eb3d4499e43?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2008&q=80",
            "https://images.unsplash.com/photo-1578927107994-75410e4dcd51?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=729&q=80",
            "https://images.unsplash.com/photo-1612760721786-a42eb89aba02?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=735&q=80",
            "https://images.unsplash.com/photo-1630476504743-a4d342f88760?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1895&q=80",
            "https://upload.wikimedia.org/wikipedia/commons/6/66/VanGogh-starry_night_ballance1.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/d/de/Van_Gogh_Starry_Night_Drawing.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg/1024px-Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Vincent_van_Gogh_-_Self-portrait_with_grey_felt_hat_-_Google_Art_Project.jpg/1024px-Vincent_van_Gogh_-_Self-portrait_with_grey_felt_hat_-_Google_Art_Project.jpg",

            "https://img.freepik.com/free-vector/young-woman-walking-dog-leash-girl-leading-pet-park-flat-illustration_74855-11306.jpg?w=996&t=st=1685117377~exp=1685117977~hmac=dd6cf9856bdac8715c1d5464875225286942da2a01ea3851ea3936dd95d96a44",
            "https://img.freepik.com/free-vector/biophilic-design-workspace-abstract-concept_335657-3081.jpg?w=996&t=st=1685117412~exp=1685118012~hmac=cc89e22bd6dbeb3c2fc06396035863e612149b04ed6dee90791292a7151a4dd2",
            "https://img.freepik.com/free-vector/pine-tree-sticker-white-background_1308-75956.jpg?w=826&t=st=1685117428~exp=1685118028~hmac=36f37f710de7b4b7320d32dc169459f0bd0d6081e94e972198ab8d0a479f67e2",
            "https://img.freepik.com/free-psd/abstract-background-design_1297-124.jpg?w=996&t=st=1685117527~exp=1685118127~hmac=08c82ea8b2087dff81e01c946f999ed6bfb286a222c09e396b4d3f46787b2b50",
            "https://images.unsplash.com/photo-1538836026403-e143e8a59f04?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1448&q=80",
            "https://images.rawpixel.com/image_1000/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvZnJydWluX3dpbmRvd19kZWNheV9sZWF2ZS1pbWFnZS1reWNmbmM5aC5qcGc.jpg",
            "https://images.unsplash.com/photo-1518562180175-34a163b1a9a6?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1740&q=80",

            "https://images.unsplash.com/photo-1654648663068-0093ade5069e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1160&q=80",
            "https://img.freepik.com/free-psd/three-dimensional-real-estate-icon-mock-up_23-2149729145.jpg?w=996&t=st=1685117577~exp=1685118177~hmac=2d789df87b156c2e5578c8ddb69e4a3b3176206f81b774d9faea7492a4eafc0f",
            "https://images.unsplash.com/photo-1644664477908-f8c4b1d215c4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2080&q=80",
            "https://images.unsplash.com/photo-1634926878768-2a5b3c42f139?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=912&q=80",
            "https://github.com/styledrop/styledrop.github.io/blob/main/images/assets/image_6487327_crayon_02.jpg",
            "https://images.unsplash.com/photo-1668090956076-b2c9d6193e6b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1935&q=80",
            "https://images.unsplash.com/photo-1637234852730-677079a9d718?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=735&q=80",
            "https://images.unsplash.com/photo-1636391891394-56a534be9a1b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1160&q=80",
        ]

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, index):
        # open image file
        url = self.urls[index]
        image = Image.open(requests.get(url, stream=True).raw)
        return image


object_prompt_list = [
    'a {unique_token} {class_token} in the jungle',
    'a {unique_token} {class_token} in the snow',
    'a {unique_token} {class_token} on the beach',
    'a {unique_token} {class_token} on a cobblestone street',
    'a {unique_token} {class_token} on top of pink fabric',
    'a {unique_token} {class_token} on top of a wooden floor',
    'a {unique_token} {class_token} with a city in the background',
    'a {unique_token} {class_token} with a mountain in the background',
    'a {unique_token} {class_token} with a blue house in the background',
    'a {unique_token} {class_token} on top of a purple rug in a forest',
    'a {unique_token} {class_token} with a wheat field in the background',
    'a {unique_token} {class_token} with a tree and autumn leaves in the background',
    'a {unique_token} {class_token} with the Eiffel Tower in the background',
    'a {unique_token} {class_token} floating on top of water',
    'a {unique_token} {class_token} floating in an ocean of milk',
    'a {unique_token} {class_token} on top of green grass with sunflowers around it',
    'a {unique_token} {class_token} on top of a mirror',
    'a {unique_token} {class_token} on top of the sidewalk in a crowded street',
    'a {unique_token} {class_token} on top of a dirt road',
    'a {unique_token} {class_token} on top of a white rug',
    'a red {unique_token} {class_token}',
    'a purple {unique_token} {class_token}',
    'a shiny {unique_token} {class_token}',
    'a wet {unique_token} {class_token}',
    'a cube shaped {unique_token} {class_token}',
]

live_prompt_list = [
    'a {unique_token} {class_token} in the jungle',
    'a {unique_token} {class_token} in the snow',
    'a {unique_token} {class_token} on the beach',
    'a {unique_token} {class_token} on a cobblestone street',
    'a {unique_token} {class_token} on top of pink fabric',
    'a {unique_token} {class_token} on top of a wooden floor',
    'a {unique_token} {class_token} with a city in the background',
    'a {unique_token} {class_token} with a mountain in the background',
    'a {unique_token} {class_token} with a blue house in the background',
    'a {unique_token} {class_token} on top of a purple rug in a forest',
    'a {unique_token} {class_token} wearing a red hat',
    'a {unique_token} {class_token} wearing a santa hat',
    'a {unique_token} {class_token} wearing a rainbow scarf',
    'a {unique_token} {class_token} wearing a black top hat and a monocle',
    'a {unique_token} {class_token} in a chef outfit',
    'a {unique_token} {class_token} in a firefighter outfit',
    'a {unique_token} {class_token} in a police outfit',
    'a {unique_token} {class_token} wearing pink glasses',
    'a {unique_token} {class_token} wearing a yellow shirt',
    'a {unique_token} {class_token} in a purple wizard outfit',
    'a red {unique_token} {class_token}',
    'a purple {unique_token} {class_token}',
    'a shiny {unique_token} {class_token}',
    'a wet {unique_token} {class_token}',
    'a cube shaped {unique_token} {class_token}',
]


class Wrapper(torch.utils.data.IterableDataset):
    def __init__(self, src_dataset, drop_last=True):
        self.source = src_dataset
        self.drop_last = drop_last
        self._count = 1
        self._seed = 0
        self._shuffle = False

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0

        mod = world_size
        shift = rank
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            mod *= worker_info.num_workers
            shift = shift * worker_info.num_workers + worker_info.id

        epoch = 0
        keys = np.arange(len(self.source))
        remainder = len(keys) % mod

        while epoch < self._count:
            if self._shuffle:
                rng = np.random.default_rng(seed=self._seed+epoch)
                rng.shuffle(keys)

            if remainder == 0:
                indices = keys
            elif self.drop_last:
                indices = keys[:-remainder]
            else:
                indices = np.concatenate((keys, keys[:mod-remainder]))

            for index in indices[shift::mod]:
                yield self.source[index]

            epoch += 1

    def repeat(self, count=float("inf")):
        self._count = count
        return self

    def shuffle(self, mode=True, seed=None):
        if isinstance(seed, int):
            self._seed = seed
        self._shuffle = mode
        return self
