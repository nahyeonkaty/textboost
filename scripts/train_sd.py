#!/usr/bin/env python3
import argparse
import importlib
import json
import logging
import os
import time
from pathlib import Path

import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm
from transformers import AutoTokenizer

from textboost.adapters import Adapter
from textboost.attention_processor import (
    build_textboost_attn_processors,
)
from textboost.datasets import SDDataset
from textboost.expand_bank import (
    ExpandAdapterBank,
    export_expand_bank_state_dict,
    iter_cross_attention_to_k_layers,
    make_expand_adapter_key,
)
from textboost.generator_builder import generator_build
from textboost.pipelines.sd import TextBoostPipeline
from textboost.text_encoders.clip import TextModel
from textboost.ti_utils import add_new_token, save_embeddings
from textboost.trainers import SDTrainer
from textboost.training_callbacks import (
    CallbackHandler,
    CheckpointCallback,
    MetricWriterCallback,
    ValidationSamplerCallback,
)

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


ARTIFACT_SCHEMA_VERSION = "textboost-artifact-v1"
UNET_CUSTOM_DIFFUSION_KV_FILENAME = "custom_diffusion_kv.bin"


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
        "--class_token",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
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
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
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
        "--skip_save_text_encoder",
        action="store_true",
        required=False,
        help="Set to not save text encoder",
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
        "--num_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=1,
        help="Rank for LoRA.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        # default=["k_proj", "v_proj"],  # best
        default=["fc2"],  # best
        help="Target modules for LoRA.",
    )
    parser.add_argument(
        "--unet_lora_rank",
        type=int,
        default=0,
        help="Legacy U-Net LoRA rank. Used when --generator_lora_rank is unset.",
    )
    parser.add_argument(
        "--generator_lora_rank",
        type=int,
        default=None,
        help=(
            "LoRA rank for generator fine-tuning when --generator_finetune=lora. "
            "If unset, falls back to --unet_lora_rank."
        ),
    )
    parser.add_argument(
        "--generator_finetune",
        type=str,
        default=None,
        choices=["none", "lora", "kv", "full"],
        help=(
            "Generator fine-tuning mode. "
            "`none`: freeze U-Net, "
            "`lora`: LoRA on attn2.to_k/to_v, "
            "`kv`: full parameter updates for attn2.to_k/to_v, "  # custom diffusion
            "`full`: full U-Net fine-tuning. "  # dreambooth
            "If unset, legacy --unet_adapter_mode/--unet_lora_rank behavior is used."
        ),
    )
    parser.add_argument(
        "--unet_adapter_mode",
        type=str,
        default="auto",
        choices=["auto", "lora_kv", "full_kv"],
        help=("Legacy U-Net adapter mode. Prefer --generator_finetune."),
    )

    parser.add_argument(
        "--max_embedding_norm",
        type=float,
        default=0.43,  # heuristically chosen.
        help="Max norm for the embedding.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="imagenet_small",
    )
    parser.add_argument(
        "--disable_cpa",
        action="store_true",
        default=False,
        help="Disable causality-preserving masking for text-encoder LoRA (naive fine-tuning).",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--expand_backend",
        type=str,
        default="legacy_unet",
        choices=["legacy_unet", "text_encoder_bank"],
        help=(
            "Expand adapter backend. "
            "`legacy_unet` keeps adapters attached to U-Net projections, "
            "`text_encoder_bank` routes expand adapters through AttentionProcessor."
        ),
    )
    parser.add_argument(
        "--expand_layer_indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of U-Net cross-attention layers to receive expand adapters.",
    )
    parser.add_argument(
        "--identifier_style",
        type=str,
        default="custom",
        choices=["custom", "ti"],
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default=None,
        help="Path to annotations JSON file (takes precedence over instance_data_dir).",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    lora_target_modules = []
    for key in args.lora_target_modules:
        if key.lower() == "q":
            lora_target_modules.append("q_proj")
        elif key.lower() == "k":
            lora_target_modules.append("k_proj")
        elif key.lower() == "v":
            lora_target_modules.append("v_proj")
        elif key.lower() == "o":
            lora_target_modules.append("out_proj")
        else:
            lora_target_modules.append(key)
    args.lora_target_modules = lora_target_modules

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
    # pipeline = DiffusionPipeline.from_pretrained(
    pipeline = TextBoostPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it.
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    pipeline.scheduler = scheduler_class.from_config(
        pipeline.scheduler.config, **scheduler_args
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
    identifier = "".join(identifier)
    if args.identifier_style != "ti" and args.class_token is not None:
        identifier = f"{identifier} {args.class_token}"

    def _format_validation_prompt(template: str, identifier_text: str) -> str:
        if "{}" in template:
            return template.format(identifier_text)
        if "<*>" in template:
            return template.replace("<*>", identifier_text)
        return template

    for validation_prompt in args.validation_prompts:
        pipeline_args = {
            "prompt": _format_validation_prompt(validation_prompt, identifier),
            "num_images_per_prompt": args.num_validation_images,
        }
        print(pipeline_args)
        with torch.autocast("cuda"):
            image = pipeline(
                **pipeline_args, num_inference_steps=25, generator=generator
            ).images
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


def adapter_artifact_dir(base_dir: str | Path, module_name: str) -> Path:
    path = Path(base_dir) / "adapter" / module_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_generator_lora_rank(args) -> int:
    if args.generator_lora_rank is not None:
        return int(args.generator_lora_rank)
    return int(args.unet_lora_rank)


def resolve_generator_finetune_mode(args) -> str:
    if args.generator_finetune is not None:
        return args.generator_finetune
    legacy_lora_rank = resolve_generator_lora_rank(args)
    if legacy_lora_rank <= 0:
        return "none"
    if args.unet_adapter_mode == "auto":
        is_custom_diffusion_pattern = (
            args.identifier_style == "custom"
            and args.lora_rank == 0
            and legacy_lora_rank == 1
        )
        return "kv" if is_custom_diffusion_pattern else "lora"
    if args.unet_adapter_mode == "lora_kv":
        return "lora"
    if args.unet_adapter_mode == "full_kv":
        return "kv"
    raise ValueError(f"Unsupported legacy unet_adapter_mode: {args.unet_adapter_mode}")


def export_unet_cross_attention_kv_state_dict(
    unet: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    state_dict = unet.state_dict()
    return {
        key: value.detach().cpu().clone().float()
        for key, value in state_dict.items()
        if ".attn2.to_k." in key or ".attn2.to_v." in key
    }


def write_checkpoint_metadata(
    checkpoint_dir: str | Path,
    *,
    step: int,
    args,
    has_text_encoder_adapter: bool,
    has_unet_lora: bool,
    has_unet_custom_diffusion_kv: bool,
    has_unet_full: bool,
    has_unet_expand_adapter: bool,
    generator_finetune: str,
    unet_expand_backend: str | None,
):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "step": int(step),
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "identifier_style": args.identifier_style,
        "cpa_enabled": not args.disable_cpa,
        "adapters": {
            "text_encoder": bool(has_text_encoder_adapter),
            "text_encoder_2": False,
            "unet_lora": bool(has_unet_lora),
            "unet_custom_diffusion_kv": bool(has_unet_custom_diffusion_kv),
            "unet_full": bool(has_unet_full),
            "generator_finetune": generator_finetune,
            "unet_expand": bool(has_unet_expand_adapter),
            "unet_expand_backend": unet_expand_backend,
        },
        "layout": {
            "text_encoder": "adapter/text_encoder/",
            "unet": "adapter/unet/",
            "unet_custom_diffusion_kv": (
                f"adapter/unet/{UNET_CUSTOM_DIFFUSION_KV_FILENAME}"
            ),
            "unet_full": "adapter/unet/",
            "text_encoder_expand_bank": "adapter/text_encoder/expand_bank.bin",
            "embeddings": ["learned_embeds.bin"],
        },
    }
    with open(checkpoint_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_arguments()

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
            with open(Path(args.output_dir) / "args.json", "w") as f:
                json.dump(vars(args), f, indent=2)
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
    logger.info(f"CPA enabled: {not args.disable_cpa}")

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = TextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )

    # Add the placeholder token in tokenizer.
    new_tokens = {}
    placeholder_token_ids = []

    new_token = add_new_token(
        tokenizer,
        text_encoder,
        args.placeholder_token,
        args.initializer_token,
    )
    token_ids = new_token.token_ids
    identifier = new_token.identifier
    placeholder_token_ids += token_ids
    args.num_vectors = new_token.num_vectors
    new_tokens[args.placeholder_token] = {
        "token_id": token_ids,
        "placeholder": new_token.placeholder,
        "num_vectors": len(token_ids),
        # "identifier": [identifier, args.class_token],
    }
    if args.identifier_style != "ti" and args.class_token is None:
        args.class_token = args.initializer_token
        logger.warning(
            "--identifier_style=%s requires class token; "
            "falling back to initializer token '%s'.",
            args.identifier_style,
            args.class_token,
        )
    if args.identifier_style == "ti":
        new_tokens[args.placeholder_token]["identifier"] = identifier
    else:
        new_tokens[args.placeholder_token]["identifier"] = [
            identifier,
            args.class_token,
        ]

    unet.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    text_encoder.requires_grad_(False)
    use_expand_bank = args.expand and args.expand_backend == "text_encoder_bank"
    if use_expand_bank and args.lora_rank <= 0:
        raise ValueError(
            "--expand_backend=text_encoder_bank requires --lora_rank > 0 "
            "for expand adapter rank."
        )
    generator_finetune = resolve_generator_finetune_mode(args)
    generator_lora_rank = resolve_generator_lora_rank(args)
    use_unet_lora = generator_finetune == "lora"
    use_unet_kv = generator_finetune == "kv"
    use_unet_full = generator_finetune == "full"
    logger.info(
        "Generator fine-tuning mode: %s (lora_rank=%s)",
        generator_finetune,
        generator_lora_rank,
    )
    expand_bank: ExpandAdapterBank | None = None
    # Add LoRA.
    if args.lora_rank > 0:
        # exclude_layers = list(range(20, 23))  # SD2.1
        exclude_layers = []
        exclude_modules = []
        if args.expand:
            exclude_layers += [22]
        for layer in exclude_layers:
            exclude_modules.append(f"layers.{layer}.self_attn.q_proj")
            exclude_modules.append(f"layers.{layer}.self_attn.k_proj")
            exclude_modules.append(f"layers.{layer}.self_attn.v_proj")
            exclude_modules.append(f"layers.{layer}.self_attn.out_proj")
            exclude_modules.append(f"layers.{layer}.mlp.fc1")
            exclude_modules.append(f"layers.{layer}.mlp.fc2")

        text_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=args.lora_target_modules,
            exclude_modules=exclude_modules,
            init_lora_weights="gaussian",
        )
        text_encoder.add_adapter(text_lora_config)
        if args.disable_cpa:
            logger.info("Added Adapter to text encoder (CPA disabled)")
        else:
            text_encoder.set_adapter_mask()
            text_encoder.replace_lora_forward()
            logger.info("Added Adapter to text encoder (CPA enabled)")

    unet, unet_generator_params = generator_build(
        unet,
        generator_finetune,
        lora_rank=generator_lora_rank,
        lora_target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "add_k_proj",
            "add_v_proj",
        ],
        kv_patterns=[".attn2.to_k.", ".attn2.to_v."],
    )

    expand = False
    if args.expand:
        selected_indices = (
            set(args.expand_layer_indices)
            if args.expand_layer_indices is not None
            else None
        )
        if use_expand_bank:
            target_layers = [
                (idx, name, module)
                for idx, name, module in iter_cross_attention_to_k_layers(unet)
                if selected_indices is None or idx in selected_indices
            ]
            expand_keys = [
                make_expand_adapter_key(name) for _, name, _ in target_layers
            ]
            expand_dims = [int(module.in_features) for _, _, module in target_layers]
            expand_bank = text_encoder.create_adapters(
                num_layers=len(target_layers),
                rank=args.lora_rank,
                keys=expand_keys,
                input_dims=expand_dims,
                bias=False,
            )
            if selected_indices is not None and not target_layers:
                logger.warning(
                    "No U-Net layers matched --expand_layer_indices=%s",
                    sorted(selected_indices),
                )
            for idx, name, _module in target_layers:
                logger.info(f"Added expand-bank adapter to U-Net layer {idx}: {name}")
            expand = len(target_layers) > 0
            logger.info(
                "Enabled text_encoder_bank expand backend with %d layers.",
                len(target_layers),
            )
        else:
            attn_to_ks = []
            for name, module in unet.named_modules():
                if "attn2.to_k" in name:
                    attn_to_ks.append((name, module))
            matched_indices = []
            for idx, (name, m) in enumerate(attn_to_ks):
                if selected_indices is not None and idx not in selected_indices:
                    continue
                adapter = Adapter(1024, args.lora_rank)
                setattr(m, "adapter", adapter)
                matched_indices.append(idx)
                expand = True
                logger.info(f"Added Adapter to U-Net layer {idx}: {name}")
            if selected_indices is not None and not matched_indices:
                logger.warning(
                    "No U-Net layers matched --expand_layer_indices=%s",
                    sorted(selected_indices),
                )
            logger.info("Added Adapter to U-Net")

    unet.set_attn_processor(build_textboost_attn_processors(unet))
    text_encoder.get_input_embeddings().requires_grad_(True)
    expand_trainable_params: list[torch.nn.Parameter] = []
    if use_expand_bank and expand_bank is not None:
        expand_trainable_params = [
            p for p in expand_bank.parameters() if p.requires_grad
        ]
    expand_param_ids = {id(p) for p in expand_trainable_params}
    num_expand_bank_params = sum(p.numel() for p in expand_trainable_params)

    num_token_params = 0
    num_te_params = 0
    num_te_adapter_params = 0
    for name, param in text_encoder.named_parameters():
        if "lora" in name:
            num_te_adapter_params += param.numel()
        elif "token_embedding" in name:
            num_token_params += param.numel()
        elif param.requires_grad:
            num_te_params += param.numel()
    num_te_adapter_params += num_expand_bank_params
    num_unet_params = 0
    num_unet_adapter_params = 0
    for name, param in unet.named_parameters():
        if id(param) in expand_param_ids:
            continue
        if "lora" in name:
            num_unet_params += param.numel()
        elif param.requires_grad:
            num_unet_adapter_params += param.numel()
    logger.info(f"Total number of token embedding parameters: {num_token_params:,}")
    logger.info(f"Total number of trainable text encoder parameters: {num_te_params:,}")
    logger.info(
        f"Total number of text encoder Adapter parameters: {num_te_adapter_params:,}"
    )
    logger.info(
        "Total number of text-side expand adapter parameters: %s",
        f"{num_expand_bank_params:,}",
    )
    logger.info(
        f"Total number of trainable U-Net parameters: {num_unet_adapter_params:,}"
    )
    logger.info(f"Total number of U-Net Adapter parameters: {num_unet_params:,}")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            unet_custom_diffusion_kv_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    if use_unet_lora:
                        unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    elif use_unet_kv:
                        unet_custom_diffusion_kv_to_save = (
                            export_unet_cross_attention_kv_state_dict(
                                unwrap_model(model)
                            )
                        )
                    elif use_unet_full:
                        unwrap_model(model).save_pretrained(
                            str(adapter_artifact_dir(output_dir, "unet"))
                        )
                elif isinstance(model, type(unwrap_model(text_encoder))):
                    (
                        unwrap_model(model).save_pretrained(
                            str(adapter_artifact_dir(output_dir, "text_encoder"))
                        )
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            if use_unet_lora:
                StableDiffusionLoraLoaderMixin.save_lora_weights(
                    str(adapter_artifact_dir(output_dir, "unet")),
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                )
            if use_unet_kv and unet_custom_diffusion_kv_to_save is not None:
                torch.save(
                    unet_custom_diffusion_kv_to_save,
                    str(
                        adapter_artifact_dir(output_dir, "unet")
                        / UNET_CUSTOM_DIFFUSION_KV_FILENAME
                    ),
                )

    accelerator.register_save_state_pre_hook(save_model_hook)

    def save_unet_custom_diffusion_artifact(output_dir: str | Path) -> None:
        if not use_unet_kv:
            return
        torch.save(
            export_unet_cross_attention_kv_state_dict(unwrap_model(unet)),
            str(
                adapter_artifact_dir(output_dir, "unet")
                / UNET_CUSTOM_DIFFUSION_KV_FILENAME
            ),
        )

    def save_unet_full_artifact(output_dir: str | Path) -> None:
        if not use_unet_full:
            return
        unwrap_model(unet).save_pretrained(
            str(adapter_artifact_dir(output_dir, "unet"))
        )

    def save_expand_artifact(output_dir: str | Path) -> None:
        if not expand:
            return
        if use_expand_bank:
            text_encoder_adapter_dir = adapter_artifact_dir(output_dir, "text_encoder")
            active_expand_bank = unwrap_model(text_encoder).get_expand_adapter_bank()
            if active_expand_bank is None:
                raise ValueError(
                    "Expand backend is text_encoder_bank, but no bank was found on text encoder."
                )
            torch.save(
                export_expand_bank_state_dict(active_expand_bank),
                str(text_encoder_adapter_dir / "expand_bank.bin"),
            )
            return

        unet_adapter_dir = adapter_artifact_dir(output_dir, "unet")
        adapter_state_dict = {}
        unet_state_dict = unwrap_model(unet).state_dict()
        for k, v in unet_state_dict.items():
            if "adapter" in k:
                adapter_state_dict[k] = v.clone().float()
        torch.save(
            adapter_state_dict,
            str(unet_adapter_dir / "adapter.bin"),
        )

    if args.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        if use_unet_lora or use_unet_kv or use_unet_full:
            unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision.
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}"
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
    text_encoder_trainable_params = list(
        filter(
            lambda p: p.requires_grad,
            text_encoder.text_model.encoder.parameters(),
        )
    )
    text_encoder_trainable_params += expand_trainable_params
    params_to_optimize = [{"params": text_encoder_trainable_params}]
    if (
        use_unet_lora
        or use_unet_kv
        or use_unet_full
        or (args.expand and not use_expand_bank)
    ):
        unet_trainable_params = [
            p for p in unet_generator_params if id(p) not in expand_param_ids
        ]
        known_param_ids = {id(p) for p in unet_trainable_params}
        unet_trainable_params += [
            p
            for p in unet.parameters()
            if p.requires_grad
            and id(p) not in expand_param_ids
            and id(p) not in known_param_ids
        ]
        params_to_optimize.append(
            {
                "params": unet_trainable_params,
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
            args.emb_lr_scheduler,
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
    train_dataset = SDDataset(
        data_path=args.instance_data_dir,
        concept_identifier=new_tokens[args.placeholder_token]["identifier"],
        tokenizer=tokenizer,
        annotations_file=args.annotations_file,
        instance=args.instance,
        num_instance=args.num_samples,
        template=args.template,
        class_token=args.class_token,
        size=args.resolution,
        center_crop=args.center_crop,
        hflip="true",
        reg_token=(args.initializer_token if args.regularization > 0 else None),
    )
    train_dataset = train_dataset.with_drop_last(False).shuffle(seed=args.seed).repeat()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=lambda examples: SDDataset.collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    if (
        use_unet_lora
        or use_unet_kv
        or use_unet_full
        or use_expand_bank
        or (args.expand and not use_expand_bank)
    ):
        (
            text_encoder,
            unet,
            emb_optimizer,
            optimizer,
            emb_lr_scheduler,
            lr_scheduler,
        ) = accelerator.prepare(
            text_encoder,
            unet,
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
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
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

    def save_checkpoint_artifacts(save_path: str, checkpoint_step: int) -> None:
        save_unet_custom_diffusion_artifact(save_path)
        save_unet_full_artifact(save_path)
        if expand:
            save_expand_artifact(save_path)
        save_embeddings(
            unwrap_model(text_encoder),
            [new_token],
            os.path.join(save_path, "learned_embeds.bin"),
            safe_serialization=False,
        )
        write_checkpoint_metadata(
            save_path,
            step=checkpoint_step,
            args=args,
            has_text_encoder_adapter=args.lora_rank > 0,
            has_unet_lora=use_unet_lora,
            has_unet_custom_diffusion_kv=use_unet_kv,
            has_unet_full=use_unet_full,
            has_unet_expand_adapter=expand,
            generator_finetune=generator_finetune,
            unet_expand_backend=(args.expand_backend if expand else None),
        )

    def run_validation_for_step(validation_step: int):
        return log_validation(
            text_encoder,
            tokenizer,
            unet,
            vae,
            args,
            accelerator,
            weight_dtype,
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
    trainer = SDTrainer(
        accelerator=accelerator,
        train_dataloader=train_dataloader,
        max_train_steps=args.max_train_steps,
        progress_bar=progress_bar,
        callbacks=callbacks,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        noise_scheduler=noise_scheduler,
        emb_optimizer=emb_optimizer,
        optimizer=optimizer,
        emb_lr_scheduler=emb_lr_scheduler,
        lr_scheduler=lr_scheduler,
        args=args,
        weight_dtype=weight_dtype,
        tokenizer=tokenizer,
        placeholder_token_ids=placeholder_token_ids,
        orig_embeds_params=orig_embeds_params,
        active_expand_bank=active_expand_bank,
        accumulate_models=(unet, text_encoder),
    )
    step = trainer.run(initial_step=initial_step)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if use_unet_lora:
            unet = unwrap_model(unet).to(torch.float32)
            unet.save_pretrained(str(adapter_artifact_dir(args.output_dir, "unet")))
        if use_unet_kv:
            save_unet_custom_diffusion_artifact(args.output_dir)
        if use_unet_full:
            save_unet_full_artifact(args.output_dir)

        if args.lora_rank > 0:
            text_encoder = unwrap_model(text_encoder).to(torch.float32)
            text_encoder.save_pretrained(
                str(adapter_artifact_dir(args.output_dir, "text_encoder"))
            )

        if expand:
            save_expand_artifact(args.output_dir)

        save_embeddings(
            unwrap_model(text_encoder),
            [new_token],
            os.path.join(args.output_dir, "learned_embeds.bin"),
            safe_serialization=False,
        )
        write_checkpoint_metadata(
            args.output_dir,
            step=args.max_train_steps,
            args=args,
            has_text_encoder_adapter=args.lora_rank > 0,
            has_unet_lora=use_unet_lora,
            has_unet_custom_diffusion_kv=use_unet_kv,
            has_unet_full=use_unet_full,
            has_unet_expand_adapter=expand,
            generator_finetune=generator_finetune,
            unet_expand_backend=(args.expand_backend if expand else None),
        )

    end_time = time.perf_counter()
    logger.info(f"Training took {end_time - start_time:.2f} seconds")
    accelerator.end_training()


if __name__ == "__main__":
    main()
