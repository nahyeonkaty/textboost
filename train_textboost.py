#!/usr/bin/env python3
import argparse
import copy
import importlib
import json
import logging
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.distributions import Categorical, Beta
from tqdm import tqdm
from transformers import AutoTokenizer

import diffusers
from diffusers import (AutoencoderKL, DDPMScheduler, DiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import (convert_state_dict_to_diffusers,
                             is_wandb_available, make_image_grid)
from diffusers.utils.torch_utils import is_compiled_module

from textboost.dataset import InstructPix2PixDataset, TextBoostDataset, PriorDataset, Wrapper
from textboost.utils import (add_augmentation_tokens, add_token, encode_prompt,
                             generate_prior_images,
                             import_model_class_from_model_name_or_path)

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def parse_args(input_args=None):
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
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance",
        type=str,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_token",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_token",
        type=str,
        nargs="+",
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_image_prior",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--image_ppl_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument("--kpl_weight", type=float, default=0.1, help="The weight of prior preservation loss.")
    parser.add_argument("--kpl_type", type=str, default="cos", help="The type of prior preservation loss.")
    parser.add_argument(
        "--num_prior_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_token."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
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
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
        "--emb_learning_rate",
        type=float,
        default=1e-3,
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_token, class_token, etc.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--skip_save_text_encoder", action="store_true", required=False, help="Set to not save text encoder"
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--validation_scheduler",
        type=str,
        default="DPMSolverMultistepScheduler",
        choices=["DPMSolverMultistepScheduler", "DDPMScheduler"],
        help="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
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
        "--unet_params_to_train",
        type=str,
        choices=["none", "crossattn_kv", "crossattn", "attn", "all"],
        default="none",
    )
    parser.add_argument(
        "--augment",
        default="none",
    )
    parser.add_argument(
        "--augment_ops",
        type=str,
        default="object",
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--augment_prompt",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--augment_inversion",
        action="store_true",
        default=False,
        help="Whether to use augment inversion.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank for LoRA.",
    )
    parser.add_argument(
        "--disable_weighted_sample",
        action="store_true",
        default=True,
        help="Whether to use weighted sampling.",
    )
    parser.add_argument(
        "--null_prob",
        type=float,
        default=0.1,
        help="Probability of null prompt.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="tepa",
    )
    parser.add_argument(
        "--mixing",
        action="store_true",
        default=False,
        help="Whether to use mixing.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.with_image_prior:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_token is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_image_prior.")
        if args.class_token is not None:
            warnings.warn("You need not use --class_token without --with_image_prior.")

    if args.augment_inversion:
        if not bool(args.augment_prompt):
            raise ValueError("You need to use --augment_prompt=1 with --augment_prompt.")

    return args


def log_validation(
    text_encoder,
    tokenizer,
    unet,
    vae,
    args,
    accelerator,
    weight_dtype,
    global_step,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompts}."
    )

    # Create pipeline (note: unet and vae are loaded again in float32).
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=accelerator.unwrap_model(text_encoder),
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.set_progress_bar_config(disable=True)

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it.
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Run inference.
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for validation_prompt in args.validation_prompts:
        for i, placeholder in enumerate(args.placeholder_token):
            placeholder_str = " ".join(placeholder)
            validation_prompt = validation_prompt.replace(f"<{i}>", placeholder_str)
        pipeline_args = {
            "prompt": validation_prompt,
            "num_images_per_prompt": args.num_validation_images,
        }
        print(pipeline_args)
        with torch.autocast("cuda"):
            image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images
        images.extend(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log({
                "validation": [
                    # wandb.Image(image, caption=f"{i}: {args.validation_prompts}")
                    wandb.Image(image, caption=f"{i}")
                    for i, image in enumerate(images)
                ]}
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def save_embedding(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids): max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

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
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_token": args.instance_token,
                "class_token": args.class_token,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir,
                "placeholder_token": args.placeholder_token,
                "initializer_token": args.initializer_token,
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    # Generate class images if prior preservation is enabled.
    if args.with_image_prior:
        class_images_dir = Path(args.class_data_dir)
        generate_prior_images(
            class_images_dir,
            args.class_token,
            args,
            logger,
            accelerator,
            use_blip_caption=True,
        )

    # Load the tokenizer.
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # Import correct text encoder class.
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    original_text_encoder = copy.deepcopy(text_encoder).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # Add the placeholder token in tokenizer.
    added_tokens = {}
    placeholder_token_dict = {}
    placeholder_token_ids = []
    args.placeholder_token = []
    args.initializer_token = []
    n = 0
    for i, concept in enumerate(args.concepts_list):
        for placeholder_token, initializer_token in zip(
            concept["placeholder_token"].split("+"), concept["initializer_token"].split("+")
        ):
            print(placeholder_token, initializer_token)
            placeholder_tokens, token_ids = add_token(
                text_encoder,
                tokenizer,
                placeholder_token,
                initializer_token,
            )
            placeholder_token_dict[n] = token_ids
            placeholder_token_ids += token_ids
            n += 1
            args.placeholder_token.append(placeholder_tokens)
            args.initializer_token.append(initializer_token)
            for token, token_id in zip(placeholder_tokens, token_ids):
                added_tokens[token] = token_id
    if args.augment_inversion:
        aug_token_ids, aug_token_dict = add_augmentation_tokens(
            text_encoder,
            tokenizer,
            aug_type="style" if args.augment_ops=="style" else "object",
        )
        added_token_ids = placeholder_token_ids + aug_token_ids
    else:
        added_token_ids = placeholder_token_ids
    print(tokenizer)
    # Update the concept list with the added tokens.
    for i, concept in enumerate(args.concepts_list):
        concept["instance_token"] = args.placeholder_token[i]
        concept["placeholder_token"] = args.placeholder_token[i]
        concept["initializer_token"] = args.initializer_token[i]

    unet.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    text_encoder.requires_grad_(False)
    # Add LoRA.
    if args.lora_rank > 0:
        text_encoder.text_model.encoder.requires_grad_(True)
        text_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        )
        text_encoder.add_adapter(text_lora_config)
        logger.info("Added LoRA to text encoder")

        if args.unet_params_to_train == "crossattn_kv":
            unet.train().requires_grad_(True)
            unet_lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["attn2.to_k", "attn2.to_v"],
            )
            unet.add_adapter(unet_lora_config)
            logger.info("Added LoRA to U-Net")
    text_encoder.get_input_embeddings().requires_grad_(True)

    num_text_params = 0
    num_lora_params = 0
    num_token_params = 0
    for name, param in text_encoder.named_parameters():
        if "lora" in name:
            num_lora_params += param.numel()
        elif "token_embedding" in name:
            num_token_params += param.numel()
        elif param.requires_grad:
            num_text_params += param.numel()
    num_unet_params = 0
    for name, param in unet.named_parameters():
        if "lora" in name:
            num_unet_params += param.numel()
        elif param.requires_grad:
            print(name)
    logger.info(f"Total number of token embedding parameters: {num_token_params:,}")
    logger.info(f"Total number of trainable text encoder parameters: {num_text_params:,}")
    logger.info(f"Total number of text encoder LoRA parameters: {num_lora_params:,}")
    logger.info(f"Total number of U-Net LoRA parameters: {num_unet_params:,}")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers = None
            text_encoder_lora_layers = None
            for model in models:
                if (
                    args.lora_rank > 0
                    and isinstance(model, type(unwrap_model(unet)))
                ):
                    unet_lora_layers = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                # elif (
                #     args.lora_rank > 0
                #     and isinstance(model, type(unwrap_model(text_encoder)))
                # ):
                #     text_encoder_lora_layers = convert_state_dict_to_diffusers(
                #         get_peft_model_state_dict(model)
                #     )

                weights.pop()

            if args.lora_rank > 0:
                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers,
                    # text_encoder_lora_layers=text_encoder_lora_layers,
                )

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(text_encoder))):
                # Load transformers style into model.
                load_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
                model.config = load_model.config
            else:
                # Load diffusers style into model.
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    # accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        if args.unet_params_to_train != "none":
            unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision.
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(f"Unet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}")

    if unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {unwrap_model(text_encoder).dtype}." f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimizer creation.
    params_to_optimize = [{
        "params": list(text_encoder.get_input_embeddings().parameters()),
        "lr": args.emb_learning_rate,
    }]
    params_to_optimize.append({
        "params": list(
            filter(lambda p: p.requires_grad, text_encoder.text_model.encoder.parameters())
        ),
    })
    if args.unet_params_to_train != "none":
        params_to_optimize.append({
            "params": list(filter(lambda p: p.requires_grad, unet.parameters())),
        })
    # Print number of parameters to optimizer.
    with torch.no_grad():
        total_params = 0
        for group in params_to_optimize:
            total_params += sum(p.numel() for p in group["params"])
        logger.info(f"Total number of parameters to optimize: {total_params:,}")
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    if args.augment in ("pda", "paug"):
        from textboost.augment import PairedAugmentation
        augment_pipe = PairedAugmentation(
            hflip="inversion" if args.augment_inversion else "false",
            augment_prompt=args.augment_prompt,
            inversion=args.augment_inversion,
            p=args.augment_p,
            ops=args.augment_ops,
        )
    elif args.augment == "custom_diff":
        from textboost.augment import CustomDiffAugment
        augment_pipe = CustomDiffAugment(size=args.resolution, hflip=True)
    else:
        augment_pipe = None

    train_dataset = TextBoostDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        num_instance=args.num_samples,
        template=args.template,
        prior_data_root=args.class_data_dir if args.with_image_prior else None,
        class_token=args.class_token,
        num_prior=args.num_prior_images,
        size=args.resolution,
        center_crop=args.center_crop,
        augment_pipe=augment_pipe,
    )
    train_dataset = Wrapper(train_dataset, drop_last=False)
    train_dataset = train_dataset.shuffle(seed=args.seed).repeat()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=lambda examples: TextBoostDataset.collate_fn(examples, args.with_image_prior),
        num_workers=args.dataloader_num_workers,
    )

    edit_dataset = InstructPix2PixDataset(tokenizer, "data/human-written-prompts.jsonl", num_samples=None)
    # print(edit_dataset)
    prior_dataset = PriorDataset(
        edit_dataset,
        tokenizer,
        additional_template="textboost",
        additional_category=args.class_token,
        null_prob=args.null_prob,
    )
    prior_dataset = Wrapper(prior_dataset, drop_last=True)
    prior_dataset = prior_dataset.shuffle(seed=args.seed).repeat()
    prior_dataloader = torch.utils.data.DataLoader(
        prior_dataset,
        batch_size=args.train_batch_size,
        collate_fn=PriorDataset.collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    if args.unet_params_to_train == "none":
        text_encoder, optimizer, lr_scheduler = accelerator.prepare(
            text_encoder, optimizer, lr_scheduler,
        )
    else:
        text_encoder, unet, optimizer, lr_scheduler = accelerator.prepare(
            text_encoder, unet, optimizer, lr_scheduler
        )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype.
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    original_text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("textboost", config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
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

    # Timestep sampling weights.
    t = torch.arange(noise_scheduler.config.num_train_timesteps, device=accelerator.device)
    logsnr = compute_snr(noise_scheduler, t).log()
    constant = logsnr.max()
    w_t = -logsnr + constant
    p_t = w_t / w_t.sum()
    dist = Categorical(probs=p_t)

    text_encoder.train()
    train_iterator = iter(train_dataloader)
    prior_iterator = iter(prior_dataloader)

    start_time = time.perf_counter()
    while step < args.max_train_steps:
        batch = next(train_iterator)

        pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
        input_ids = batch["input_ids"].to(accelerator.device)
        attention_mask = batch["attention_mask"]

        prior_batch = next(prior_iterator)
        prior_input_ids = prior_batch["input_ids"].to(accelerator.device)
        prior_attention_mask = prior_batch["attention_mask"]

        # Convert images to latent space
        model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor

        with accelerator.accumulate(unet, text_encoder):
            # Sample noise that we'll add to the model input.
            noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image.
            if args.disable_weighted_sample:
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=model_input.device,
                )
            else:
                timesteps = dist.sample((bsz,)).to(accelerator.device)
            # Add noise to the model input according to the noise magnitude at each timestep.
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
            # Get the text embedding for conditioning.
            encoder_hidden_states = encode_prompt(
                text_encoder,
                input_ids,
                attention_mask,
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
            )

            # Predict the noise residual.
            model_pred = unet(
                noisy_model_input.to(unet.dtype),
                timesteps,
                encoder_hidden_states.to(unet.dtype),
            ).sample

            # Get the target for loss depending on the prediction type.
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.with_image_prior:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                # Compute prior loss.
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Compute instance loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            if "mask" in batch:
                mask = batch["mask"].to(accelerator.device, dtype=weight_dtype)
                loss = ((loss*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])).mean()
            else:
                loss = loss.mean()

            if args.with_image_prior:
                # Add the prior loss to the instance loss.
                loss = loss + args.image_ppl_weight * prior_loss

            if args.kpl_weight > 0.0:
                prior_attention_mask = torch.cat(prior_attention_mask, dim=0).to(accelerator.device)
                prior_hidden_states = text_encoder(prior_input_ids)[0].float()
                original_prior_hidden_states = original_text_encoder(prior_input_ids)[0].float()
                if args.kpl_type == 'cos':
                    kp_loss = 1 - F.cosine_similarity(prior_hidden_states, original_prior_hidden_states, dim=-1)
                    kp_loss = kp_loss.mean()
                else:
                    kp_loss = F.mse_loss(prior_hidden_states, original_prior_hidden_states, reduction="mean")
                loss = loss + args.kpl_weight * kp_loss

            accelerator.backward(loss)
            if args.placeholder_token:
                if hasattr(text_encoder, "module"):
                    grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                index_grads_to_zero = torch.arange(len(tokenizer)) < min(added_token_ids)
                grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                    index_grads_to_zero, :
                ].fill_(0)

            if args.mixing:
                for name, param in text_encoder.named_parameters():
                    if "lora_B" in name:
                        if args.augment_ops == "object":
                            # zero out the gradients or odd indices
                            param.grad[1::2, :] = 0.0
                        else:
                            param.grad[0::2, :] = 0.0

            if accelerator.sync_gradients:
                params_to_clip = (
                    accelerator.unwrap_model(text_encoder)
                    .text_model.encoder.parameters()
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)


        # Checks if the accelerator has performed an optimization step behind the scenes.
        if accelerator.sync_gradients:
            progress_bar.update(1)
            step += 1

            if accelerator.is_main_process:
                if step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)
                    save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                    accelerator.save_state(save_path)
                    (
                        accelerator.unwrap_model(text_encoder)
                        .to(torch.float32)
                        .save_pretrained(os.path.join(save_path, "text_encoder"))
                    )
                    logger.info(f"Saved state to {save_path}")

                    # Save the embeddings.
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    for token, token_id in added_tokens.items():
                        learned_embeds = (
                            accelerator.unwrap_model(text_encoder)
                            .get_input_embeddings()
                            .weight[token_id]
                        )
                        learned_embeds_dict = {token: learned_embeds.detach().cpu()}
                        token = token.replace("<", "").replace(">", "")
                        save_path = os.path.join(ckpt_dir, f"{token}.bin")
                        torch.save(learned_embeds_dict, save_path)

                    if args.augment_inversion:
                        for token, token_id in aug_token_dict.items():
                            learned_embeds = (
                                accelerator.unwrap_model(text_encoder)
                                .get_input_embeddings()
                                .weight[token_id : token_id + 1]
                            )
                            learned_embeds_dict = {token: learned_embeds.detach().cpu()}
                            token = token.replace("<", "").replace(">", "")
                            save_path = os.path.join(ckpt_dir, f"{token}.bin")
                            torch.save(learned_embeds_dict, save_path)

                images = []

                if args.validation_prompts and step % args.validation_steps == 0:
                    images = log_validation(
                        text_encoder,
                        tokenizer,
                        unet,
                        vae,
                        args,
                        accelerator,
                        weight_dtype,
                        step,
                    )
                    if images:
                        rows = len(args.validation_prompts)
                        cols = args.num_validation_images
                        image_grid = make_image_grid(images, rows, cols)
                        image_grid.save(os.path.join(args.output_dir, f"validation_{step}.jpg"))

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.unet_params_to_train != "none":
            unet = unwrap_model(unet).to(torch.float32)
            unet.save_pretrained(os.path.join(args.output_dir, "unet"))

        if args.lora_rank > 0:
            text_encoder = unwrap_model(text_encoder).to(torch.float32)
            text_encoder.save_pretrained(os.path.join(args.output_dir, "text_encoder"))

        for token, token_id in added_tokens.items():
            learned_embeds = (
                accelerator.unwrap_model(text_encoder)
                .get_input_embeddings()
                .weight[token_id]
            )
            learned_embeds_dict = {token: learned_embeds.detach().cpu()}
            token = token.replace("<", "").replace(">", "")
            save_path = os.path.join(args.output_dir, f"{token}.bin")
            torch.save(learned_embeds_dict, save_path)

        if args.augment_inversion:
            for token, token_id in aug_token_dict.items():
                learned_embeds = (
                    accelerator.unwrap_model(text_encoder)
                    .get_input_embeddings()
                    .weight[token_id : token_id + 1]
                )
                learned_embeds_dict = {token: learned_embeds.detach().cpu()}
                token = token.replace("<", "").replace(">", "")
                save_path = os.path.join(args.output_dir, f"{token}.bin")
                torch.save(learned_embeds_dict, save_path)

    end_time = time.perf_counter()
    logger.info(f"Training took {end_time - start_time:.2f} seconds")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
