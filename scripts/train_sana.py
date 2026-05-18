#!/usr/bin/env python3
"""
TextBoost SANA Training Script

This script supports two dataset formats using the unified SanaDataset interface:

1. Legacy directory-based format (--data_dir):
   Use a directory containing images for a single instance.

   Example:
   python scripts/train_sana.py \
       --pretrained_model_name_or_path="Efficient-Large-Model/Sana_1600M_1024px_diffusers" \
       --data_dir="datasets/my_instance" \
       --placeholder_token="<myobj>" \
       --initializer_token="object" \
       --class_token="object" \
       --output_dir="./output/myobj"

2. Annotations-based format (--annotations_file + --instance):
   Use a JSON file containing grouped instances with descriptions.
   The unified interface now supports custom descriptions from annotations
   while maintaining template-based fallbacks.

   Example:
   python scripts/train_sana.py \
       --pretrained_model_name_or_path="Efficient-Large-Model/Sana_1600M_1024px_diffusers" \
       --annotations_file="datasets/dataset_annotations.json" \
       --instance="backpack" \
       --placeholder_token="<backpack>" \
       --initializer_token="backpack" \
       --class_token="backpack" \
       --output_dir="./output/backpack"

The unified SanaDataset automatically handles:
- Custom descriptions from annotations (when available)
- Template-based prompt generation (as fallback)
- Smart prompt mixing (50% custom vs template when both available)
- Support for both grouped and flat annotation formats
"""

import argparse
import copy
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from transformers import AutoTokenizer

import diffusers
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict

from textboost.adapters import Adapter, TrfConfig, attach_adapters_to_model
from textboost.attention_processor import build_textboost_attn_processors
from textboost.datasets import SanaDataset
from textboost.expand_bank import (
    ExpandAdapterBank,
    export_expand_bank_state_dict,
    iter_cross_attention_to_k_layers,
    make_expand_adapter_key,
)
from textboost.generator_builder import generator_build
from textboost.pipelines.sana import TextBoostSanaPipeline
from textboost.text_encoders.gemma2 import TextModel as Gemma2TextModel
from textboost.training_callbacks import (
    CallbackHandler,
    CheckpointCallback,
    MetricWriterCallback,
    ValidationSamplerCallback,
)
from textboost.trainers import SanaTrainer
from textboost.ti_utils import (
    add_new_token,
    save_embeddings,
)

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


TRANSFORMER_CUSTOM_DIFFUSION_KV_FILENAME = "custom_diffusion_kv.bin"


def adapter_artifact_dir(base_dir: str | Path, module_name: str) -> Path:
    path = Path(base_dir) / "adapter" / module_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _collect_cross_attention_kv_prefixes(dit: torch.nn.Module) -> list[str]:
    prefixes: set[str] = set()
    for _idx, name, _module in iter_cross_attention_to_k_layers(dit):
        prefixes.add(name)
        if ".to_k" in name:
            prefixes.add(name.replace(".to_k", ".to_v"))
    if not prefixes:
        for name, _module in dit.named_modules():
            if ".to_k" in name:
                prefixes.add(name)
                prefixes.add(name.replace(".to_k", ".to_v"))
    return sorted(prefixes)


def export_transformer_cross_attention_kv_state_dict(
    dit: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    prefixes = _collect_cross_attention_kv_prefixes(dit)
    state_dict = dit.state_dict()
    return {
        key: value.detach().cpu().clone().float()
        for key, value in state_dict.items()
        if any(key.startswith(f"{prefix}.") for prefix in prefixes)
    }


def parse_arguments(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default=None,
        help="Path to JSON file containing dataset annotations (instance-grouped format). If provided, takes precedence over --data_dir.",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Instance name to use when loading from annotations_file. Required when using --annotations_file.",
    )
    parser.add_argument(
        "--class_token",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=300,
        help="Maximum sequence length to use with with the Gemma model",
    )
    parser.add_argument(
        "--complex_human_instruction",
        type=str,
        default="\n".join(
            [
                "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
                "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
                "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
                "Here are examples of how to transform or refine prompts:",
                "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
                "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
                "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
                "User Prompt: ",
            ]
        ),
        help="Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.",
    )
    parser.add_argument(
        "--chi_prob",
        type=float,
        default=0.0,
        help="Probability of using complex human instruction for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--emb_learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--emb_lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=(
            'We default to the "none" weighting scheme for uniform sampling and uniform loss'
        ),
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompts` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
    )

    # Textual Inversion.
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<dog>",
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default="dog",
        help="A token to use as initializer word.",
    )
    parser.add_argument(
        "--max_embedding_norm",
        type=float,
        # default=2.5,  # heuristically chosen.
        default=4.86,  # max value of the pretrained embeddings.
        help="Max norm for the embedding.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="imagenet_small",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--expand_backend",
        type=str,
        default="text_encoder_bank",
        choices=["text_encoder_bank", "dit_adapter"],
        help=(
            "TextBoost++ backend. `text_encoder_bank` stores expand adapters on "
            "text encoder (recommended). `dit_adapter` keeps legacy DiT-attached adapters."
        ),
    )
    parser.add_argument(
        "--identifier_style",
        type=str,
        default="ti",
        choices=["custom", "ti"],
    )

    # Fine-tuning.
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=1,
        help="Rank for LoRA.",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["down_proj"],
        help="Target modules for LoRA.",
    )
    parser.add_argument(
        "--generator_finetune",
        type=str,
        default="none",
        choices=["none", "lora", "kv", "full"],
        help=(
            "Generator (DiT) fine-tuning mode. "
            "`none`: freeze transformer, "
            "`lora`: LoRA adapters on transformer modules, "
            "`kv`: full-parameter updates for cross-attn to_k/to_v only, "
            "`full`: full transformer fine-tuning."
        ),
    )
    parser.add_argument(
        "--generator_lora_rank",
        type=int,
        default=4,
        help="LoRA rank for generator fine-tuning when --generator_finetune=lora.",
    )
    parser.add_argument(
        "--generator_target_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
        help="Generator LoRA target modules for --generator_finetune=lora.",
    )

    # Accelerator (and hardware settings).
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    target_modules = []
    for key in args.target_modules:
        if key.lower() == "q":
            target_modules.append("q_proj")
        elif key.lower() == "k":
            target_modules.append("k_proj")
        elif key.lower() == "v":
            target_modules.append("v_proj")
        elif key.lower() == "o":
            target_modules.append("o_proj")
        else:
            target_modules.append(key)
    args.target_modules = target_modules

    generator_target_modules = []
    for key in args.generator_target_modules:
        key_l = key.lower()
        if key_l == "q":
            generator_target_modules.append("to_q")
        elif key_l == "k":
            generator_target_modules.append("to_k")
        elif key_l == "v":
            generator_target_modules.append("to_v")
        elif key_l == "o":
            generator_target_modules.append("to_out.0")
        else:
            generator_target_modules.append(key)
    args.generator_target_modules = generator_target_modules

    return args


def log_validation(
    tokenizer,
    text_encoder,
    vae,
    transformer,
    args,
    accelerator,
    global_step,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompts}."
    )

    # Create pipeline (note: unet and vae are loaded again in float32).
    # pipeline = DiffusionPipeline.from_pretrained(
    # pipeline = TextBoostSanaPipeline.from_pretrained(
    pipeline = TextBoostSanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=accelerator.unwrap_model(text_encoder),
        vae=vae,
        transformer=accelerator.unwrap_model(transformer),
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Run inference.
    generator = (
        None
        if args.seed is None
        else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    )
    images = []
    identifier = [args.placeholder_token]
    if args.num_vectors > 1:
        identifier[0] = identifier[0].replace(">", "_0>")
        for i in range(1, args.num_vectors):
            identifier.append(args.placeholder_token.replace(">", f"_{i}>"))
    identifier = "".join(identifier)  # Gemma tokenizer also tokenize (' ').
    for validation_prompt in args.validation_prompts:
        pipeline_args = {
            "prompt": validation_prompt.format(identifier),
            "num_images_per_prompt": args.num_validation_images,
        }
        print(pipeline_args)
        # with torch.autocast("cuda"):
        #     image = pipeline(**pipeline_args, generator=generator).images
        image = pipeline(**pipeline_args, generator=generator).images
        images.extend(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, global_step, dataformats="NHWC"
            )
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        # wandb.Image(image, caption=f"{i}: {args.validation_prompts}")
                        wandb.Image(image, caption=f"{i}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def main():
    args = parse_arguments()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    if args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=Path(args.output_dir, "training.log"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is None:
        args.seed = np.random.randint(1 << 31)
    logger.info(f"Using random seed: {args.seed}")
    set_seed(args.seed)

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # Load scheduler and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder = Gemma2TextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderDC.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    dit = SanaTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )

    # Add the placeholder token in tokenizer.
    new_tokens = []
    added_token_ids = []

    new_token = add_new_token(
        tokenizer,
        text_encoder,
        args.placeholder_token,
        args.initializer_token,
        joiner="",
    )
    args.num_vectors = new_token.num_vectors
    if args.identifier_style == "ti":
        new_token.concept_identifier = new_token.identifier
    else:
        new_token.concept_identifier = [new_token.identifier, args.class_token]
    added_token_ids += new_token.token_ids
    new_tokens.append(new_token)

    # print(tokenizer)
    print(added_token_ids)
    print(new_tokens)

    if args.expand and args.lora_rank <= 0:
        raise ValueError("`--expand` requires `--lora_rank > 0`.")

    generator_finetune = args.generator_finetune
    use_dit_lora = generator_finetune == "lora"
    use_dit_kv = generator_finetune == "kv"
    use_dit_full = generator_finetune == "full"
    if use_dit_lora and args.generator_lora_rank <= 0:
        raise ValueError(
            "--generator_finetune=lora requires --generator_lora_rank > 0."
        )

    dit.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    text_encoder.requires_grad_(False)
    # Add LoRA.
    if args.lora_rank > 0:
        exclude_modules = []
        # exclude_layers = [0, 1] + list(range(20, 26))
        # for l in exclude_layers:
        #     exclude_modules.append(f"layers.{l}.self_attn.q_proj")
        #     exclude_modules.append(f"layers.{l}.self_attn.k_proj")
        #     exclude_modules.append(f"layers.{l}.self_attn.v_proj")
        #     exclude_modules.append(f"layers.{l}.self_attn.o_proj")
        #     exclude_modules.append(f"layers.{l}.mlp.down_proj")
        trf_config = TrfConfig(
            r=args.lora_rank,
            target_index=added_token_ids,
            target_modules=args.target_modules,
            exclude_modules=exclude_modules,
        )
        text_encoder = attach_adapters_to_model(text_encoder, trf_config)
        logger.info("Added Adapter to text encoder")

    dit, dit_generator_params = generator_build(
        dit,
        generator_finetune,
        lora_rank=args.generator_lora_rank,
        lora_target_modules=args.generator_target_modules,
    )
    logger.info(
        "Applied generator fine-tuning mode to DiT: %s (lora_rank=%s, target_modules=%s)",
        generator_finetune,
        args.generator_lora_rank,
        args.generator_target_modules,
    )

    expand = bool(args.expand)
    use_expand_bank = expand and args.expand_backend == "text_encoder_bank"
    expand_bank: ExpandAdapterBank | None = None
    if expand:
        dit.set_attn_processor(build_textboost_attn_processors(dit, use_sana=True))
        if use_expand_bank:
            cross_attention_layers = list(iter_cross_attention_to_k_layers(dit))
            expand_keys = [
                make_expand_adapter_key(name) for _, name, _ in cross_attention_layers
            ]
            expand_dims = [
                int(module.in_features) for _, _, module in cross_attention_layers
            ]
            expand_bank = text_encoder.create_adapters(
                num_layers=len(cross_attention_layers),
                rank=args.lora_rank,
                keys=expand_keys,
                input_dims=expand_dims,
                bias=False,
            )
            logger.info(
                "Added text-encoder expand adapter bank for %s SANA cross-attn layers.",
                len(cross_attention_layers),
            )
        else:
            for _index, _name, module in iter_cross_attention_to_k_layers(dit):
                adapter = Adapter(int(module.in_features), args.lora_rank)
                setattr(module, "adapter", adapter)
            logger.info("Added legacy DiT-attached expand adapters.")
    text_encoder.get_input_embeddings().requires_grad_(True)

    num_token_params = 0
    num_te_params = 0
    num_te_adapter_params = 0
    num_expand_bank_params = 0
    for name, param in text_encoder.named_parameters():
        if "lora" in name:
            num_te_adapter_params += param.numel()
        elif "_expand_adapter_bank" in name:
            num_expand_bank_params += param.numel()
        elif "embed_tokens" in name:
            num_token_params += param.numel()
        elif param.requires_grad:
            num_te_params += param.numel()
    num_te_adapter_params += num_expand_bank_params
    num_dit_params = 0
    num_unet_adapter_params = 0
    for name, param in dit.named_parameters():
        if "lora" in name:
            num_dit_params += param.numel()
        elif param.requires_grad:
            num_unet_adapter_params += param.numel()
    logger.info(f"Total number of token embedding parameters: {num_token_params:,}")
    logger.info(f"Total number of trainable text encoder parameters: {num_te_params:,}")
    logger.info(
        f"Total number of text encoder Adapter parameters: {num_te_adapter_params:,}"
    )
    if num_expand_bank_params > 0:
        logger.info(
            "Total number of text encoder expand-bank parameters: %s",
            f"{num_expand_bank_params:,}",
        )
    logger.info(
        f"Total number of trainable DiT parameters: {num_unet_adapter_params:,}"
    )
    logger.info(f"Total number of DiT Adapter parameters: {num_dit_params:,}")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    text_encoding_pipeline = TextBoostSanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=None,
        transformer=None,
    )

    if args.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        if (use_dit_lora or use_dit_kv or use_dit_full) and hasattr(
            dit, "enable_gradient_checkpointing"
        ):
            dit.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision.
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if unwrap_model(dit).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {unwrap_model(dit).dtype}. {low_precision_error_string}"
        )

    if unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Optimizer creation.
    text_encoder_trainable_params = [
        param
        for name, param in text_encoder.named_parameters()
        if param.requires_grad and "embed_tokens" not in name
    ]
    params_to_optimize = [
        {
            "params": text_encoder_trainable_params,
        }
    ]
    if use_dit_lora or use_dit_kv or use_dit_full or (expand and not use_expand_bank):
        dit_trainable_params = list(dit_generator_params)
        known_param_ids = {id(p) for p in dit_trainable_params}
        dit_trainable_params += [
            param
            for param in dit.parameters()
            if param.requires_grad and id(param) not in known_param_ids
        ]
        params_to_optimize.append(
            {
                "params": dit_trainable_params,
            }
        )
    # Print number of parameters to optimizer.
    with torch.no_grad():
        total_params = 0
        for group in params_to_optimize:
            total_params += sum(p.numel() for p in group["params"])
        logger.info(f"Total number of parameters to optimize: {total_params:,}")

    emb_optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=args.emb_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if args.emb_lr_scheduler == "cosine":
        emb_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            emb_optimizer,
            T_max=args.max_train_steps,
            eta_min=args.learning_rate,
        )
    else:
        emb_lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=emb_optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Dataset and DataLoaders creation:
    # Create dataset using unified interface
    # The refactored SanaDataset now supports both data_path and annotations_file modes
    # Benefits:
    # - Single class handles both traditional and annotation-based datasets
    # - Smart prompt generation: custom descriptions + template fallbacks
    # - Unified parameter interface reduces code complexity
    # - Enhanced flexibility with description-based training
    if args.annotations_file is not None:
        # Use annotations-based dataset with custom descriptions
        if args.instance is None:
            raise ValueError("--instance is required when using --annotations_file")
        train_dataset = SanaDataset(
            annotations_file=args.annotations_file,
            instance=args.instance,
            concept_identifier=new_token.concept_identifier,
            num_instance=args.num_samples,
            template=args.template,
            size=args.resolution,
            center_crop=args.center_crop,
        )
    else:
        # Use legacy directory-based dataset with template prompts
        if args.data_dir is None:
            raise ValueError("Either --annotations_file or --data_dir must be provided")
        train_dataset = SanaDataset(
            data_path=args.data_dir,
            concept_identifier=new_token.concept_identifier,
            num_instance=args.num_samples,
            template=args.template,
            size=args.resolution,
            center_crop=args.center_crop,
        )

    # Log dataset configuration and refactoring benefits
    dataset_mode = (
        "annotations-based" if args.annotations_file is not None else "directory-based"
    )
    logger.info(f"Created unified SanaDataset in {dataset_mode} mode")
    if args.annotations_file is not None:
        logger.info(f"  Using annotations file: {args.annotations_file}")
        logger.info(f"  Target instance: {args.instance}")
        logger.info(
            "  Benefits: Custom descriptions + template fallbacks, enhanced prompt diversity"
        )
    else:
        logger.info(f"  Using data directory: {args.data_dir}")
        logger.info("  Benefits: Template-based prompts with consistent formatting")

    train_dataset = train_dataset.with_drop_last(False).shuffle(seed=args.seed).repeat()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    if use_dit_lora or use_dit_kv or use_dit_full or expand:
        (
            text_encoder,
            dit,
            emb_optimizer,
            optimizer,
            emb_lr_scheduler,
            lr_scheduler,
        ) = accelerator.prepare(
            text_encoder,
            dit,
            emb_optimizer,
            optimizer,
            emb_lr_scheduler,
            lr_scheduler,
        )
    else:
        (
            text_encoder,
            emb_optimizer,
            optimizer,
            emb_lr_scheduler,
            lr_scheduler,
        ) = accelerator.prepare(
            text_encoder,
            emb_optimizer,
            optimizer,
            emb_lr_scheduler,
            lr_scheduler,
        )

    active_expand_bank: ExpandAdapterBank | None = None
    if use_expand_bank and expand:
        active_expand_bank = unwrap_model(text_encoder).get_expand_adapter_bank()
        if active_expand_bank is None:
            raise ValueError(
                "Expand backend is text_encoder_bank, but no bank was found on text encoder."
            )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype.
    # NOTE: fp16 is recommended for the transformer.
    # https://huggingface.co/docs/diffusers/en/api/pipelines/sana
    if (
        args.pretrained_model_name_or_path
        == "Efficient-Large-Model/Sana_600M_512px_diffusers"
    ):
        dit_dtype = torch.float16
    else:
        dit_dtype = torch.bfloat16
    dit.to(accelerator.device, dtype=dit_dtype)
    for name, module in dit.named_modules():
        if "adapter" in name:
            module.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use.
    if accelerator.is_main_process:
        accelerator.init_trackers("textboost")
        if args.report_to == "wandb":
            wandb.watch(text_encoder)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    step = 0
    initial_step = 0

    # Potentially load in the weights and states from a previous save.
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            step = int(path.split("-")[1])

            initial_step = step

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def save_transformer_lora_artifact(output_dir: str | Path) -> None:
        if not use_dit_lora:
            return
        lora_state = get_peft_model_state_dict(unwrap_model(dit))
        TextBoostSanaPipeline.save_lora_weights(
            str(adapter_artifact_dir(output_dir, "transformer")),
            transformer_lora_layers=lora_state,
        )

    def save_transformer_kv_artifact(output_dir: str | Path) -> None:
        if not use_dit_kv:
            return
        torch.save(
            export_transformer_cross_attention_kv_state_dict(unwrap_model(dit)),
            str(
                adapter_artifact_dir(output_dir, "transformer")
                / TRANSFORMER_CUSTOM_DIFFUSION_KV_FILENAME
            ),
        )

    def save_transformer_full_artifact(output_dir: str | Path) -> None:
        if not use_dit_full:
            return
        unwrap_model(dit).save_pretrained(
            str(adapter_artifact_dir(output_dir, "transformer"))
        )

    def save_checkpoint_artifacts(save_path: str, checkpoint_step: int) -> None:
        del checkpoint_step
        if args.lora_rank > 0:
            state_dict = unwrap_model(text_encoder).state_dict()
            torch.save(state_dict, Path(save_path) / "text_encoder.bin")
            with open(Path(save_path) / "config.json", "w") as f:
                json.dump(asdict(trf_config), f, indent=2)

        save_transformer_lora_artifact(save_path)
        save_transformer_kv_artifact(save_path)
        save_transformer_full_artifact(save_path)

        if expand:
            if use_expand_bank:
                checkpoint_expand_bank = unwrap_model(
                    text_encoder
                ).get_expand_adapter_bank()
                if checkpoint_expand_bank is None:
                    raise ValueError(
                        "Expand backend is text_encoder_bank, but no bank was found on text encoder."
                    )
                torch.save(
                    export_expand_bank_state_dict(checkpoint_expand_bank),
                    Path(save_path) / "expand_bank.bin",
                )
            else:
                adapter_state_dict = {}
                unet_state_dict = unwrap_model(dit).state_dict()
                for k, v in unet_state_dict.items():
                    if "adapter" in k:
                        adapter_state_dict[k] = v.clone().float()
                torch.save(adapter_state_dict, os.path.join(save_path, "adapter.bin"))

        save_embeddings(
            accelerator.unwrap_model(text_encoder),
            new_tokens,
            os.path.join(save_path, "learned_embeds.bin"),
            safe_serialization=False,
        )

    def run_validation_for_step(validation_step: int):
        return log_validation(
            tokenizer,
            text_encoder,
            vae,
            dit,
            args,
            accelerator,
            validation_step,
        )

    callbacks = CallbackHandler(
        [
            MetricWriterCallback(accelerator, progress_bar),
            CheckpointCallback(
                accelerator=accelerator,
                output_dir=args.output_dir,
                checkpointing_steps=args.checkpointing_steps,
                checkpoints_total_limit=args.checkpoints_total_limit,
                logger=logger,
                save_artifact_fn=save_checkpoint_artifacts,
            ),
            ValidationSamplerCallback(
                accelerator=accelerator,
                validation_steps=args.validation_steps,
                validation_prompts=args.validation_prompts,
                num_validation_images=args.num_validation_images,
                output_dir=args.output_dir,
                run_validation_fn=run_validation_for_step,
            ),
        ]
    )

    text_encoder.train()

    # keep original embeddings as reference
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight.data.clone()
    )

    start_time = time.perf_counter()
    trainer = SanaTrainer(
        accelerator=accelerator,
        train_dataloader=train_dataloader,
        max_train_steps=args.max_train_steps,
        progress_bar=progress_bar,
        callbacks=callbacks,
        dit=dit,
        text_encoder=text_encoder,
        vae=vae,
        text_encoding_pipeline=text_encoding_pipeline,
        noise_scheduler_copy=noise_scheduler_copy,
        emb_optimizer=emb_optimizer,
        optimizer=optimizer,
        emb_lr_scheduler=emb_lr_scheduler,
        lr_scheduler=lr_scheduler,
        args=args,
        dit_dtype=dit_dtype,
        tokenizer=tokenizer,
        added_token_ids=added_token_ids,
        orig_embeds_params=orig_embeds_params,
        use_attention_kwargs=expand,
        accumulate_models=(dit, text_encoder),
    )
    step = trainer.run(initial_step=initial_step)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.lora_rank > 0:
            text_encoder = unwrap_model(text_encoder).to(torch.float32)
            save_path = Path(args.output_dir) / "text_encoder"
            save_path.mkdir(parents=True, exist_ok=True)
            state_dict = unwrap_model(text_encoder).state_dict()
            torch.save(state_dict, save_path / "text_encoder.bin")
            with open(save_path / "config.json", "w") as f:
                json.dump(asdict(trf_config), f, indent=2)

        save_transformer_lora_artifact(args.output_dir)
        save_transformer_kv_artifact(args.output_dir)
        save_transformer_full_artifact(args.output_dir)

        if expand:
            if use_expand_bank:
                final_expand_bank = unwrap_model(text_encoder).get_expand_adapter_bank()
                if final_expand_bank is None:
                    raise ValueError(
                        "Expand backend is text_encoder_bank, but no bank was found on text encoder."
                    )
                torch.save(
                    export_expand_bank_state_dict(final_expand_bank),
                    os.path.join(args.output_dir, "expand_bank.bin"),
                )
            else:
                adapter_state_dict = {}
                unet_state_dict = unwrap_model(dit).state_dict()
                for k, v in unet_state_dict.items():
                    if "adapter" in k:
                        adapter_state_dict[k] = v.clone().float()
                torch.save(
                    adapter_state_dict,
                    os.path.join(args.output_dir, "adapter.bin"),
                )

        save_embeddings(
            accelerator.unwrap_model(text_encoder),
            new_tokens,
            os.path.join(args.output_dir, "learned_embeds.bin"),
            safe_serialization=False,
        )

    end_time = time.perf_counter()
    logger.info(f"Training took {end_time - start_time:.2f} seconds")
    accelerator.end_training()


if __name__ == "__main__":
    main()
