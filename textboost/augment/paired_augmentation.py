import math
import random

import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
from PIL import Image
from torchvision.transforms import v2


def _compute_padding(h, w, scale):
    # (pad_h + h) * scale = h
    # (pad_w + w) * scale = w
    pad_h = round(((h / scale) - h) / 2)
    pad_w = round(((w / scale) - w) / 2)
    return pad_h, pad_w


def adjust_scale(image, prompt, inversion=False):
    # random scale in range [0.34, 1.4]
    scale_factor = np.random.uniform(0.34, 1.4)
    h, w = image.size
    pad_h, pad_w = _compute_padding(h, w, scale_factor)
    if pad_h > 0 and pad_w > 0:
        image = v2.functional.pad(image, (pad_w, pad_h), padding_mode="edge")
    image = v2.functional.affine(
        image,
        angle=0, translate=(0, 0), scale=scale_factor, shear=0,
        interpolation=Image.BICUBIC,
    )
    image = v2.functional.center_crop(image, (h, w))
    if inversion:
        if scale_factor < 0.6:
            add_to_caption = "<zoom-out_0> <zoom-out_1>"
        elif scale_factor > 1.2:
            add_to_caption = "<zoom-in_0> <zoom-in_1>"
        else:
            add_to_caption = ""
        prompt = add_to_caption + prompt
    else:
        if scale_factor <= 0.6:
            add_to_caption = np.random.choice(["a far away ", "very small "])
        elif scale_factor >= 1.2:
            add_to_caption = np.random.choice(["zoomed in ", "close up "])
        else:
            add_to_caption = ""
        prompt = add_to_caption + prompt
    return image, prompt


def rotate(image, prompt, inversion=False):
    rot_direction = np.random.randint(0, 2)
    if inversion:
        if rot_direction == 0:
            image = v2.functional.rotate(image, angle=90)
            add_to_caption = "<rot90_0> <rot90_1>"
        elif rot_direction == 1:
            image = v2.functional.rotate(image, angle=-90)
            add_to_caption = "<rot270_0> <rot270_1>"
        if np.random.rand() < 0.5:
            prompt = add_to_caption + prompt
        else:
            prompt = prompt + ", " + add_to_caption
    else:
        if rot_direction == 0:
            image = v2.functional.rotate(image, angle=90)
            add_to_caption = "90 degrees counter clockwise rotated "
        elif rot_direction == 1:
            image = v2.functional.rotate(image, angle=-90)
            add_to_caption = "90 degrees clockwise rotated "
        prompt = add_to_caption + prompt
    return image, prompt


def horizontal_flip(image, prompt, inversion=False):
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if inversion:
        if np.random.rand() < 0.5:
            prompt = "<hflip> " + prompt
        else:
            prompt = prompt + ", <hflip>"
    else:
        if np.random.rand() < 0.5:
            prompt = "horizontally flipped " + prompt
        else:
            prompt = prompt + ", horizontally flipped"
    return image, prompt


def horizontal_translate(image, prompt, inversion=False):
    shift_dir = np.random.randint(0, 2)
    w, h = image.size

    shift_str = np.random.uniform(low=0.15, high=0.3)
    shift = int(shift_str * w)
    if inversion:
        if shift_dir == 0:
            trans = [-shift, 0]
            padding = (shift, 0)
            add_to_caption = " <left_0> <left_1> <left_2>"
        else:
            trans = [shift, 0]
            padding = (shift, 0)
            add_to_caption = " <right_0> <right_0> <right_0>"
        prompt = prompt + add_to_caption
    else:
        if shift_dir == 0:
            trans = [-shift, 0]
            padding = (shift, 0)
            add_to_caption = " on the left"
        else:
            trans = [shift, 0]
            padding = (shift, 0)
            add_to_caption = " on the right"
        prompt = prompt + add_to_caption
    image = v2.functional.pad(image, padding, padding_mode="edge")
    image = v2.functional.affine(
        image,
        angle=0,
        translate=trans,
        scale=1,
        shear=0,
    )
    image = v2.functional.center_crop(image, [w, h])
    return image, prompt


def adjust_brightness(image, prompt, inversion=False, size=None):
    _ = size  # unused
    if np.random.random() < 0.5:
        factor = np.random.uniform(0.4, 0.6)
        if inversion:
            add_to_caption = "<dimmed>"
        else:
            add_to_caption = "dimmed"
    else:
        factor = np.random.uniform(1.3, 1.5)
        if inversion:
            add_to_caption = "<bright>"
        else:
            add_to_caption = "bright"

    enhancer = PIL.ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(factor)
    if np.random.random() < 0.5:
        prompt = add_to_caption + prompt
    else:
        prompt = prompt + f", {add_to_caption}"
    return enhanced_image, prompt


def grayscale(image, prompt, inversion=False, size=None):
    _ = size  # unused
    image = PIL.ImageOps.grayscale(image).convert("RGB")

    if inversion:
        add_to_prompt = "<grayscale_0> <grayscale_1>"
    else:
        add_to_prompt = "grayscale"

    prompt = f"{prompt}, {add_to_prompt}"
    return image, prompt


def random_resized_crop(image, target_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
    """
    Performs a random resized crop on an image.

    Args:
        image: A PIL Image object.
        target_size: A tuple representing the desired output size (width, height).
        scale: A tuple representing the minimum and maximum scaling factors.
        ratio: A tuple representing the minimum and maximum aspect ratios.

    Returns:
        A PIL Image object that has been randomly cropped and resized to the target size.
    """

    width, height = image.size
    target_width, target_height = target_size

    # Generate random area and aspect ratio within the specified ranges
    area = width * height * random.uniform(*scale)
    aspect_ratio = random.uniform(*ratio)

    # Calculate new width and height based on area and aspect ratio
    new_width = int(round(math.sqrt(area * aspect_ratio)))
    new_height = int(round(math.sqrt(area / aspect_ratio)))

    # Ensure new dimensions don't exceed original image dimensions
    if new_width > width:
        new_width = width
    if new_height > height:
        new_height = height

    # Calculate random crop coordinates
    x = random.randint(0, width - new_width)
    y = random.randint(0, height - new_height)

    # Crop and resize the image
    cropped_image = image.crop((x, y, x + new_width, y + new_height))
    resized_image = cropped_image.resize(target_size, Image.BICUBIC)

    return resized_image


def crop(image, prompt, inversion=False):
    h, w = image.size
    image = random_resized_crop(image, (h, w), ratio=(1.0, 1.0))

    if inversion:
        add_to_prompt = "<crop>"
    else:
        add_to_prompt = "cropped"

    if np.random.random() < 0.5:
        prompt = f"{add_to_prompt} {prompt}"
    else:
        prompt = f"{prompt}, {add_to_prompt}"
    return image, prompt


def jpeg_compression(image, prompt, inversion=False):
    quality = np.random.randint(25, 75)
    image = v2.functional.jpeg(image, quality=quality)

    if inversion:
        add_to_prompt = f"<jpeg_0> <jpeg_1>"
    else:
        add_to_prompt = f"JPEG"

    if np.random.random() < 0.5:
        prompt = f"{add_to_prompt} {prompt}"
    else:
        prompt = f"{prompt}, {add_to_prompt}"
    return image, prompt


def square_photo_collage(image, prompt, inversion=False):
    axis = np.random.choice([2, 4])
    w, h = image.size
    grid_w, w_remainder = divmod(w, axis)
    grid_h, h_remainder = divmod(h, axis)
    small_image = np.asarray(image.resize((grid_h, grid_w), Image.BICUBIC))

    grid = np.zeros([grid_w * axis, grid_h * axis, 3], dtype=np.uint8)
    # print(grid.shape, small_image.shape)
    for i in range(0, grid.shape[0], grid_w):
        for j in range(0, grid.shape[1], grid_h):
            grid[i:i + grid_w, j:j + grid_h] = small_image
    image = Image.fromarray(grid)

    if inversion:
        prompt = "<grid_0> <grid_1> " + prompt
    else:
        prompt = "grid of " + prompt
    return image, prompt


def cutout(image, prompt, inversion=False):
    image = np.asarray(image).copy()
    h, w, c = image.shape
    cutout_ratio = np.random.uniform(0.1, 0.33)
    cutout_size = round(h * cutout_ratio)
    x = np.random.randint(0, w - cutout_size)
    y = np.random.randint(0, h - cutout_size)
    image[y:y + cutout_size, x:x + cutout_size] = 0
    image = PIL.Image.fromarray(image)
    if inversion:
        if np.random.rand() < 0.5:
            prompt = "<cutout> " + prompt
        else:
            prompt = prompt + ", <cutout>"
    else:
        prompt = "cutout " + prompt
    return image, prompt


class PairedAugmentation:
    def __init__(
            self,
            hflip="false",
            inversion=False,
            p=0.5,
            color_prob=0.5,
            augment_prompt=True,
            ops="object",
        ):
        assert hflip.lower() in ("true", "false", "inversion"), \
            f"Invalid hflip value: {hflip}"
        self.hflip = False
        self.inversion = inversion
        self.p = p
        self.color_prob = color_prob
        self.augment_prompt = augment_prompt

        if ops == "object":
            self.geometric_ops = [
                adjust_scale,
                crop,
                horizontal_translate,
                # rotate,
            ]
            self.color_ops = [
                grayscale,
                adjust_brightness,
                # jpeg_compression,
            ]
            self.other_ops = [
                cutout,
                square_photo_collage,
            ]
            # self.ops = [
            #     grayscale,
            #     adjust_scale,
            #     adjust_brightness,
            #     crop,
            #     horizontal_translate,
            #     cutout,
            #     grid,
            # ]
        else:  # "style"
            self.geometric_ops = []
            self.color_ops = [grayscale]
            self.other_ops = []
            # self.ops = [
            #     horizontal_flip,
            # ]
        if hflip == "inversion":
            # self.ops.append(horizontal_flip)
            self.geometric_ops.append(horizontal_flip)
        elif hflip == "true":
            self.hflip = True

    def __call__(self, image, prompt):
        assert isinstance(image, PIL.Image.Image), \
            f"Invalid image type ({type(image)}). Must be PIL.Image.Image."

        if self.hflip and np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # if len(self.ops) > 0 and np.random.rand() < self.p:
        #     op = np.random.choice(self.ops)
        #     image, new_prompt = op(image, prompt, self.inversion)
        #     if self.augment_prompt:
        #         prompt = new_prompt

        if len(self.geometric_ops) > 0 and np.random.rand() < self.p:
            op = np.random.choice(self.geometric_ops)
            image, new_prompt = op(image, prompt, self.inversion)
            if self.augment_prompt:
                prompt = new_prompt

        if len(self.color_ops) > 0 and np.random.rand() < self.color_prob:
            op = np.random.choice(self.color_ops)
            image, new_prompt = op(image, prompt, self.inversion)
            if self.augment_prompt:
                prompt = new_prompt

        if len(self.other_ops) > 0 and np.random.rand() < self.p:
            op = np.random.choice(self.other_ops)
            image, new_prompt = op(image, prompt, self.inversion)
            if self.augment_prompt:
                prompt = new_prompt

        return image, prompt, None
