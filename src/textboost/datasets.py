from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision.transforms import v2

IMAGENET_TEMPLATES_SMALL = [
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

IMAGENET_STYLE_TEMPLATES_SMALL = [
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

TEXTBOOST_TEMPLATES = [
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
    ## additional
    "a zoomed in photo of the {}",
    "a zoomed in photo of a {}",
    "a far away photo of the {}",
    "a far away photo of a {}",
    "a very small photo of the {}",
    "a very small photo of a {}",
]


def tokenize_prompt(tokenizer, prompt: str, tokenizer_max_length: int | None = None):
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


def is_image_file(file_path: str | Path) -> bool:
    file_path = str(file_path).lower()
    extensions = list(Image.registered_extensions().keys())
    return any(file_path.endswith(ext) for ext in extensions)


def get_images_path(
    data_root: str | Path, max_samples: int | None = None
) -> list[Path]:
    if is_image_file(data_root):
        return [data_root]

    data_root = Path(data_root)
    if not data_root.exists():
        raise ValueError("Data root doesn't exists.")
    images_path = list(data_root.iterdir())
    images_path.sort()
    if max_samples is not None:
        return images_path[:max_samples]
    print("Number of samples: ", len(images_path))
    return images_path


class BaseDataset(torch.utils.data.IterableDataset):
    """Base dataset class with common functionality for TextBoost datasets."""

    def __init__(
        self,
        data_path=None,
        concept_identifier=None,
        annotations_file=None,
        instance=None,
        num_instance: int | None = None,
        template: str = "a {}",
        class_token: str | None = None,
        size: int = 512,
        center_crop: bool = False,
        hflip: bool = False,
    ):
        # Validate input parameters
        if annotations_file is not None:
            if instance is None:
                raise ValueError(
                    "instance parameter is required when using annotations_file"
                )
            if concept_identifier is None:
                raise ValueError(
                    "concept_identifier parameter is required when using annotations_file"
                )
        elif data_path is not None:
            if concept_identifier is None:
                raise ValueError(
                    "concept_identifier parameter is required when using data_path"
                )
        else:
            raise ValueError("Either data_path or annotations_file must be provided")

        self.size = size
        self.center_crop = center_crop
        self.class_token = class_token
        self.annotations_file = annotations_file
        self.instance = instance
        self.concept_identifier = concept_identifier

        # Setup image transforms
        self.hflip = hflip
        self._setup_transforms()

        # Setup samples - to be implemented by subclasses
        self.samples = []
        self._setup_samples(data_path, concept_identifier, num_instance)

        # Handle template selection
        self._setup_templates(template)

        self.num_instance_images = len(self.samples)
        self._length = self.num_instance_images
        self._count = float("inf")
        self._seed = 0
        self._shuffle = False
        self._drop_last = False

    def _setup_templates(self, template: str) -> None:
        """Setup prompt templates."""
        try:
            self.template = {
                "imagenet_small": IMAGENET_TEMPLATES_SMALL,
                "imagenet_style_small": IMAGENET_STYLE_TEMPLATES_SMALL,
                "textboost": TEXTBOOST_TEMPLATES,
            }[template]
        except (KeyError, TypeError):
            self.template = [template]

    def _setup_transforms(self) -> None:
        """Setup image transforms."""
        self.resize_fn = v2.Resize(
            int(self.size * 1.2), interpolation=v2.InterpolationMode.LANCZOS
        )
        self.crop = (
            v2.CenterCrop(self.size) if self.center_crop else v2.RandomCrop(self.size)
        )
        self.image_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Random crop setup
        crop = v2.RandomResizedCrop(self.size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
        prob = 0.5
        self.rand_crop = v2.Lambda(lambda x: crop(x) if random.random() < prob else x)

    def _setup_samples(self, data_path, concept_identifier, num_instance) -> None:
        """Setup samples - handles both data_path and annotations_file approaches."""
        if self.annotations_file is not None:
            self._setup_samples_from_annotations(concept_identifier, num_instance)
        else:
            self._setup_samples_from_data_path(
                data_path, concept_identifier, num_instance
            )

    def _setup_samples_from_data_path(
        self, data_path, concept_identifier, num_instance
    ) -> None:
        """Setup samples from data_path - to be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement _setup_samples_from_data_path"
        )

    def _setup_samples_from_annotations(self, concept_identifier, num_instance) -> None:
        """Setup samples from annotations file."""
        with open(self.annotations_file, "r") as f:
            annotations_data = json.load(f)

        instance_name = self.instance
        print(f"Looking for instance: '{instance_name}' in annotations")

        instance_data = annotations_data.get(instance_name)
        if instance_data is None:
            available = sorted(annotations_data.keys())
            raise ValueError(
                f"Instance '{instance_name}' not found in annotations file: {self.annotations_file}. "
                f"Available instances (first 10): {available[:10]}"
            )

        # Supported schemas:
        # 1) grouped: {"images":[{"image_path":..., "description":...}, ...], ...}
        # 2) one-shot flat: {"path":".../00.jpg", "class":"...", "initialization":"..."}
        if isinstance(instance_data, dict) and "images" in instance_data:
            images = instance_data.get("images", [])
            if num_instance is not None:
                images = images[:num_instance]

            for data in images:
                image_path = data.get("image_path") or data.get("path")
                description = data.get("description")
                if image_path is None:
                    continue
                # Keep concept identifier explicit for SD/SDXL; optional description can be used by SanaDataset.
                if description is None:
                    self.samples.append((image_path, concept_identifier))
                else:
                    self.samples.append((image_path, concept_identifier, description))
        elif isinstance(instance_data, dict):
            image_path = instance_data.get("path") or instance_data.get("image_path")
            if image_path is None:
                raise ValueError(
                    f"Unsupported annotation schema for '{instance_name}'. "
                    "Expected key 'images' or 'path'."
                )
            for p in get_images_path(image_path, max_samples=num_instance):
                self.samples.append((str(p), concept_identifier))
        else:
            raise ValueError(
                f"Unsupported annotation type for '{instance_name}': {type(instance_data)}"
            )

        if not self.samples:
            raise ValueError(
                f"No samples found for instance '{instance_name}' in annotations file {self.annotations_file}"
            )

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        if self._length == 0:
            return

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
        keys = np.arange(self._length)
        remainder = self._length % mod

        while epoch < self._count:
            if self._shuffle:
                rng = np.random.default_rng(seed=self._seed + epoch)
                rng.shuffle(keys)

            if remainder == 0:
                indices = keys
            elif self._drop_last:
                indices = keys[:-remainder]
            else:
                indices = np.concatenate((keys, keys[: mod - remainder]))

            for index in indices[shift::mod]:
                yield self[int(index)]
            epoch += 1

    def repeat(self, count=float("inf")):
        self._count = count
        return self

    def shuffle(self, mode=True, seed=None):
        if isinstance(seed, int):
            self._seed = seed
        self._shuffle = mode
        return self

    def with_drop_last(self, drop_last: bool = True):
        self._drop_last = drop_last
        return self

    def _load_and_preprocess_image(self, image_path) -> Image.Image:
        """Load and preprocess image."""
        image = Image.open(image_path)
        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image

    def _apply_horizontal_flip(self, image: Image.Image) -> Image.Image:
        """Apply horizontal flip with concept identifier modification if needed."""
        if self.hflip and np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def _generate_prompt(
        self, concept_identifier: str | list[str], description: str | None = None
    ) -> str:
        """Generate prompt from concept identifier and optional description."""
        if isinstance(concept_identifier, list):
            concept_identifier = " ".join(
                [str(token) for token in concept_identifier if token is not None]
            ).strip()

        if description is not None:
            prompt = description
        else:
            prompt_idx = random.randint(0, len(self.template) - 1)
            prompt = self.template[prompt_idx]
        prompt = prompt.format(concept_identifier)
        return prompt

    def _resize_and_crop_image(
        self, image: Image.Image
    ) -> tuple[Image.Image, int, int]:
        """Resize and crop image."""
        image = self.resize_fn(image)
        if self.center_crop:
            crop_top = max(0, int(round((image.height - self.size) / 2.0)))
            crop_left = max(0, int(round((image.width - self.size) / 2.0)))
            image = self.crop(image)
        else:
            crop_top, crop_left, h, w = self.crop.get_params(
                image, (self.size, self.size)
            )
            image = v2.functional.crop(image, crop_top, crop_left, h, w)
        return image, crop_top, crop_left


class SDDataset(BaseDataset):
    def __init__(
        self,
        data_path=None,
        concept_identifier=None,
        tokenizer=None,
        tokenizer_2=None,
        annotations_file=None,
        instance=None,
        num_instance=None,
        template="a {}",
        class_token=None,
        size=512,
        center_crop=False,
        hflip: bool = False,
        reg_token=None,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.reg_token = reg_token

        super().__init__(
            data_path=data_path,
            concept_identifier=concept_identifier,
            annotations_file=annotations_file,
            instance=instance,
            num_instance=num_instance,
            template=template,
            class_token=class_token,
            size=size,
            center_crop=center_crop,
            hflip=hflip,
        )

    def _setup_samples_from_data_path(
        self, data_path, concept_identifier, num_instance
    ):
        """Setup samples for SDDataset."""
        images_path = [
            (x, concept_identifier) for x in get_images_path(data_path, num_instance)
        ]
        self.samples.extend(images_path)

    def __getitem__(self, index):
        sample = {}

        sample_data = self.samples[index % self.num_instance_images]
        image_path, concept_identifier = sample_data[0], sample_data[1]
        if isinstance(concept_identifier, list):
            concept_identifier = " ".join(
                [str(token) for token in concept_identifier if token is not None]
            ).strip()

        image = self._load_and_preprocess_image(image_path)

        # Horizontal flip
        image = self._apply_horizontal_flip(image)

        # Generate prompt (without adding period randomly for this dataset)
        prompt_idx = random.randint(0, len(self.template) - 1)
        prompt = self.template[prompt_idx].format(concept_identifier)

        image = self.rand_crop(image)

        sample["original_size"] = (image.width, image.height)
        image, crop_top, crop_left = self._resize_and_crop_image(image)
        sample["image"] = self.image_transforms(image)  # remaining transforms.
        sample["crop_top_left"] = (crop_top, crop_left)

        text_inputs = tokenize_prompt(self.tokenizer, prompt)
        sample["input_ids"] = text_inputs.input_ids
        sample["attention_mask"] = text_inputs.attention_mask
        if self.tokenizer_2 is not None:
            text_inputs_2 = tokenize_prompt(self.tokenizer_2, prompt)
            sample["input_ids_2"] = text_inputs_2.input_ids
            sample["attention_mask_2"] = text_inputs_2.attention_mask

        # Adapter mask.
        adapter_mask = text_inputs.attention_mask.clone()
        unknown_tokens = (text_inputs.input_ids > 49407).float()
        first_unknown_token = torch.argmax(unknown_tokens)
        adapter_mask[:, :first_unknown_token] = 0.0
        sample["adapter_mask"] = adapter_mask
        if self.reg_token is not None:
            reg_prompt = self.template[prompt_idx].format(self.reg_token)
            reg_text_inputs = tokenize_prompt(self.tokenizer, reg_prompt)
            sample["reg_input_ids"] = reg_text_inputs.input_ids
        return sample

    @staticmethod
    def collate_fn(samples):
        has_attention_mask = "attention_mask" in samples[0]

        input_ids = [sample["input_ids"] for sample in samples]
        pixel_values = [sample["image"] for sample in samples]
        adapter_mask = [sample["adapter_mask"] for sample in samples]

        if has_attention_mask:
            attention_mask = [example["attention_mask"] for example in samples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)

        adapter_mask = torch.stack(adapter_mask)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "adapter_mask": adapter_mask,
        }

        if "mask" in samples[0]:
            mask = [sample["mask"] for sample in samples]
            if "prior_mask" in samples[0]:
                mask += [sample["prior_mask"] for sample in samples]
            batch["mask"] = torch.stack(mask)

        if has_attention_mask:
            batch["attention_mask"] = attention_mask

        if "reg_input_ids" in samples[0].keys():
            reg_input_ids = [sample["reg_input_ids"] for sample in samples]
            reg_input_ids = torch.cat(reg_input_ids, dim=0)
            batch["reg_input_ids"] = reg_input_ids

        return batch


class SDXLDataset(BaseDataset):
    def __init__(
        self,
        data_path,
        concept_identifier,
        tokenizer,
        tokenizer_2=None,
        num_instance=None,
        template: str = "a {}",
        size: int = 512,
        center_crop: bool = False,
        hflip: bool = False,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        super().__init__(
            data_path=data_path,
            concept_identifier=concept_identifier,
            num_instance=num_instance,
            template=template,
            size=size,
            center_crop=center_crop,
            hflip=hflip,
        )

        # Override resize function for SDXL (different from base)
        self.resize_fn = v2.Resize(size, interpolation=v2.InterpolationMode.LANCZOS)

    def _setup_samples_from_data_path(
        self, data_path, concept_identifier, num_instance
    ):
        """Setup samples for SDXLDataset."""
        images_path = [
            (x, concept_identifier) for x in get_images_path(data_path, num_instance)
        ]
        self.instance_images_path = images_path
        self.samples = images_path

    def __getitem__(self, index):
        sample = {}

        sample_data = self.instance_images_path[index % self.num_instance_images]
        image_path, concept_identifier = sample_data[0], sample_data[1]
        if isinstance(concept_identifier, list):
            concept_identifier = " ".join(
                [str(token) for token in concept_identifier if token is not None]
            ).strip()

        image = self._load_and_preprocess_image(image_path)

        # Horizontal flip (simplified for SDXL)
        if self.hflip and np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        prompt = self._generate_prompt(concept_identifier)

        sample["original_size"] = (image.width, image.height)
        image, crop_top, crop_left = self._resize_and_crop_image(image)
        sample["image"] = self.image_transforms(image)  # remaining transforms.
        sample["crop_top_left"] = (crop_top, crop_left)

        text_inputs = tokenize_prompt(self.tokenizer, prompt)
        sample["input_ids"] = text_inputs.input_ids
        sample["attention_mask"] = text_inputs.attention_mask

        if self.tokenizer_2 is not None:
            text_inputs_2 = tokenize_prompt(self.tokenizer_2, prompt)
            sample["input_ids_2"] = text_inputs_2.input_ids
            sample["attention_mask_2"] = text_inputs_2.attention_mask

        return sample


class SanaDataset(BaseDataset):
    def __init__(
        self,
        data_path=None,
        concept_identifier=None,
        annotations_file=None,
        instance=None,
        num_instance: int | None = None,
        template: str = "a {}",
        size: int = 512,
        center_crop: bool = False,
        hflip: bool = False,
    ):
        super().__init__(
            data_path=data_path,
            concept_identifier=concept_identifier,
            annotations_file=annotations_file,
            instance=instance,
            num_instance=num_instance,
            template=template,
            size=size,
            center_crop=center_crop,
            hflip=hflip,
        )

    def _setup_samples_from_data_path(
        self, data_path, concept_identifier, num_instance
    ):
        """Setup samples for SanaDataset."""
        for image_path in get_images_path(data_path, num_instance):
            if is_image_file(image_path):
                self.samples.append((image_path, concept_identifier))

    def __getitem__(self, index):
        sample = {}

        # Handle both 2-tuple (data_path mode) and 3-tuple (annotations mode)
        sample_data = self.samples[index % self.num_instance_images]
        if len(sample_data) >= 3:
            image_path = sample_data[0]
            description = sample_data[2]
        else:
            image_path = sample_data[0]
            description = None

        image = self._load_and_preprocess_image(image_path)

        # Horizontal flip.
        image = self._apply_horizontal_flip(image)

        prompt = self._generate_prompt(self.concept_identifier, description)
        sample["prompt"] = prompt

        image = self.rand_crop(image)

        image, _, _ = self._resize_and_crop_image(image)
        sample["pixel_values"] = self.image_transforms(image)  # remaining transforms.
        return sample
