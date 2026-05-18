#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_class_map(reference_path: str = "datasets/dreambooth_n1_sana.json") -> dict:
    """Load instance->class mapping from a reference one-shot manifest."""
    path = Path(reference_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    mapping = {}
    for instance, meta in data.items():
        if isinstance(meta, dict) and "class" in meta:
            mapping[instance] = meta["class"]
    return mapping


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create dataset annotations using Qwen-2.5-VL"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/dreambooth",
        help="Root directory containing instance subdirectories with images",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="datasets/qwen_annotations.json",
        help="Path to save the generated dataset annotations JSON file",
    )
    parser.add_argument(
        "--style",
        action="store_true",
        help="If set, use style descriptions instead of object descriptions",
    )
    args = parser.parse_args()
    return args


def get_init_token(processor, model, images, text):
    if not isinstance(images, list):
        images = [images]
    # content = [{"type": "image", "image": str(image)} for image in images]
    content = [{"type": "image", "image": str(images[0])}]
    # print(text)
    content.append({"type": "text", "text": text})
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


def get_description(processor, model, image, text):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


def process_dreambooth(
    processor,
    model,
    data_root,
    class_map=None,
):
    if class_map is None:
        class_map = {}
    prompt = "\n".join(
        [
            "Identify the main object in the image and describe it in 1-2 words.",
            "RULES:",
            "- Focus ONLY on the object's type, notable features, and material.",
            "- DO NOT describe the object's action, pose, or position (e.g., avoid 'sitting' or 'running').",
            "- DO NOT describe the background.",
            "- DO NOT describe colors unless they are not common for the object.",
            "- Use adjective-noun format where possible.",
            "EXAMPLES:",
            "- Input:  -> Output: 'rubber duck'",
            "- Input:  -> Output: 'corgi'",
        ]
    )
    # prompt = "\n".join(
    #     [
    #         "Identify the main object in the image and describe it in 1-3 words.",
    #         "RULES:",
    #         "- Focus ONLY on the object's type, notable features, and material.",
    #         "- DO NOT describe the object's action, pose, or position (e.g., avoid 'sitting' or 'running').",
    #         "- DO NOT describe the background.",
    #         "- DO NOT describe colors unless they are not common for the object.",
    #         "- Use adjective-noun format where possible.",
    #         "EXAMPLES:",
    #         "- Input:  -> Output: 'rubber duck'",
    #         "- Input:  -> Output: 'corgi'",
    #     ]
    # )
    instance_to_init = {}
    instances_dir = sorted(Path(data_root).iterdir())
    for instance_dir in instances_dir:
        if not instance_dir.is_dir():
            continue

        images = sorted(instance_dir.glob("*.jpg"))
        if not images:
            print(f"No images found in {instance_dir}.")
            continue

        tokens = get_init_token(
            processor,
            model,
            images,
            prompt,
        )
        name = " ".join(tokens).replace(".", "").strip()
        instance = instance_dir.name
        instance_to_init[instance] = name
        print(f"{instance}: {instance_to_init[instance]}")

    images = sorted(Path(data_root).rglob("*.jpg"))
    if not images:
        print("No images found in the specified directory.")
        return

    # Group images by instance.
    instances_data = {}

    for image in images:
        instance_name = image.parent.name
        init_token = instance_to_init.get(instance_name, "object")

        if instance_name not in instances_data:
            instances_data[instance_name] = {
                "instance": instance_name,
                "class": class_map.get(instance_name, instance_name),
                "init_token": init_token,
                "images": [],
            }

        # Generate description for this specific image.
        # text = "Describe the object in this image. Do not describe the background."
        text = "\n".join(
            [
                "Describe this image in one sentence."
                "Ignore the features (such as label, color, texture) of the object itself."
                f"Please start with 'A {init_token}'"
            ]
        )
        description = get_description(
            processor,
            model,
            str(image),
            text=text,
        )
        description = description[0].strip()
        description = description.replace(init_token, "{}")

        image_data = {"image_path": str(image), "description": description}

        instances_data[instance_name]["images"].append(image_data)
        print(f"{image.name}: {init_token} - {image_data['description']}")
    return instance_to_init, instances_data


def process_styledrop(
    processor,
    model,
    data_root,
    class_map=None,
):
    if class_map is None:
        class_map = {}
    text = "\n".join(
        [
            "Generate a 3-word description of the image's artistic style. Focus on the style, medium, and mood. Ignore the image's subject matter.",
            "Examples:",
            "- 'Vibrant cartoon illustration'",
            "- 'Minimalist line sketch'",
            "- 'Realistic oil painting'",
        ]
    )
    instance_to_init = {}
    instances_dir = sorted(Path(data_root).iterdir())
    for image_file in instances_dir:
        if not image_file.is_file():
            continue

        tokens = get_init_token(
            processor,
            model,
            image_file,
            text,
        )
        name = " ".join(tokens).replace(".", "").strip()
        instance = image_file.stem
        instance_to_init[instance] = name
        print(f"{instance}: {instance_to_init[instance]}")

    images = sorted(Path(data_root).rglob("*.png"))
    if not images:
        print("No images found in the specified directory.")
        return

    # Group images by instance.
    instances_data = {}

    for image in images:
        # instance_name = file name without extension
        instance_name = image.stem
        init_token = instance_to_init.get(instance_name, "style")

        if instance_name not in instances_data:
            instances_data[instance_name] = {
                "instance": instance_name,
                "class": class_map.get(instance_name, "style"),
                "init_token": init_token,
                "images": [],
            }

        # Generate description for this specific image.
        prompt = (
            "You are an expert image captioner. Your task is to describe the main subject and action of the image in a single, concise sentence. "
            "The sentence MUST conclude with the provided artistic style. Do not mention the style anywhere else.\n\n"
            f"Structure: [One-sentence description of image content] in a {init_token} style.\n\n"
            "Example: 'A blue robot happily waves its hand in a vibrant cartoon illustration style.'\n\n"
            "Now, describe this image:"
        )
        description = get_description(
            processor,
            model,
            str(image),
            text=prompt,
        )
        description = description[0].strip()
        description = description.replace(init_token, "{}")

        image_data = {"image_path": str(image), "description": description}

        instances_data[instance_name]["images"].append(image_data)
        print(f"{image.name}: {init_token} - {image_data['description']}")
    return instance_to_init, instances_data


def main():
    args = parse_arguments()
    class_map = load_class_map()

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving,
    #  especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    if not args.style:
        print("Processing DreamBooth dataset...")
        instance_to_init, dataset = process_dreambooth(
            processor,
            model,
            args.data_root,
            class_map=class_map,
        )
    else:
        instance_to_init, dataset = process_styledrop(
            processor,
            model,
            args.data_root,
            class_map=class_map,
        )

    print(dataset)
    # {
    #   "instance_name": {
    #       "instance": "instance_name",
    #       "init_token": "generated_token",
    #       "images": [
    #           "image_path": "path/to/image1.jpg",
    #           "description": "A {} with ..."
    #           ...

    # Save the grouped data as JSON.
    output_file = Path(args.output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    total_images = sum(len(v["images"]) for v in dataset.values())
    print(f"\nDataset saved to {output_file}")
    print(f"Total instances: {len(dataset)}")
    print(f"Total images processed: {total_images}")


if __name__ == "__main__":
    main()
