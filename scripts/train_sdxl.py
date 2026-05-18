#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm
from transformers import CLIPTokenizer

from textboost.adapters import Adapter
from textboost.attention_processor import (
    build_textboost_attn_processors,
)
from textboost.datasets import SDXLDataset
from textboost.expand_bank import (
    ExpandAdapterBank,
    export_expand_bank_state_dict,
    iter_cross_attention_to_k_layers,
    make_expand_adapter_key,
)
from textboost.generator_builder import generator_build
from textboost.pipelines.sdxl import TextBoostSDXLPipeline
from textboost.text_encoders.clip import (
    TextModel as CLIPTextModel,
)
from textboost.text_encoders.clip import (
    TextModelWithProjection as CLIPTextModelWithProjection,
)
from textboost.training_callbacks import (
    CallbackHandler,
    CheckpointCallback,
    MetricWriterCallback,
    PooledEmbeddingTrackerCallback,
    ValidationSamplerCallback,
)
from textboost.trainers import SDXLTrainer
from textboost.ti_utils import (
    add_new_token,
    save_embeddings,
)

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


ARTIFACT_SCHEMA_VERSION = "textboost-artifact-v1"
UNET_CUSTOM_DIFFUSION_KV_FILENAME = "custom_diffusion_kv.bin"


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
    has_text_encoder_2_adapter: bool,
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
        "identifier_style": "custom",
        "cpa_enabled": True,
        "adapters": {
            "text_encoder": bool(has_text_encoder_adapter),
            "text_encoder_2": bool(has_text_encoder_2_adapter),
            "unet_lora": bool(has_unet_lora),
            "unet_custom_diffusion_kv": bool(has_unet_custom_diffusion_kv),
            "unet_full": bool(has_unet_full),
            "generator_finetune": generator_finetune,
            "unet_expand": bool(has_unet_expand_adapter),
            "unet_expand_backend": unet_expand_backend,
        },
        "layout": {
            "text_encoder": "adapter/text_encoder/",
            "text_encoder_2": "adapter/text_encoder_2/",
            "unet": "adapter/unet/",
            "unet_custom_diffusion_kv": (
                f"adapter/unet/{UNET_CUSTOM_DIFFUSION_KV_FILENAME}"
            ),
            "unet_full": "adapter/unet/",
            "text_encoder_2_expand_bank": "adapter/text_encoder_2/expand_bank.bin",
            "embeddings": ["learned_embeds.bin", "learned_embeds_2.bin"],
        },
    }
    with open(checkpoint_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


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
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_token",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--do_edm_style_training",
        default=False,
        action="store_true",
        help="Flag to conduct training using the EDM formulation as introduced in https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
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
        default=768,
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
        "--adapter_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--unet_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
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
        "--track_pooled_embeddings",
        action="store_true",
        help=(
            "Track SDXL pooled embedding adapter effect (text_encoder_2) across "
            "training and save per-prompt comparison logs."
        ),
    )
    parser.add_argument(
        "--pooled_embedding_log_steps",
        type=int,
        default=0,
        help=(
            "Log pooled embedding drift every N training steps. "
            "If <= 0, uses --validation_steps."
        ),
    )
    parser.add_argument(
        "--pooled_embedding_prompts",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Prompt templates/prompts to track pooled embeddings for. "
            "Use '{}' or '<*>' as identifier placeholders. "
            "If omitted, all prompts from the selected training template are used."
        ),
    )
    parser.add_argument(
        "--pooled_embedding_batch_size",
        type=int,
        default=32,
        help="Batch size for pooled embedding tracking forward passes.",
    )
    parser.add_argument(
        "--save_pooled_embedding_vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save raw pooled embedding tensors for each logged step under "
            "output_dir/pooled_embeddings."
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
        default=2,
        help="Rank for LoRA.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="*",
        default=["fc2"],
        help="Target modules for LoRA.",
    )
    parser.add_argument(
        "--lora_target_modules_2",
        type=str,
        nargs="*",
        default=["fc2"],
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
            "`kv`: full parameter updates for attn2.to_k/to_v, "
            "`full`: full U-Net fine-tuning. "
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
        default=0.48,
        help="Max norm for the embeddings.",
    )
    parser.add_argument(
        "--max_embedding_norm_2",
        type=float,
        default=0.74,
        help="Max norm for the embeddings.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="imagenet_small",
    )
    parser.add_argument(
        "--identifier_style",
        type=str,
        default="custom",
        choices=["custom", "ti", "ours"],
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

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    lora_target_modules_2 = []
    for key in args.lora_target_modules_2:
        if key.lower() == "q":
            lora_target_modules_2.append("q_proj")
        elif key.lower() == "k":
            lora_target_modules_2.append("k_proj")
        elif key.lower() == "v":
            lora_target_modules_2.append("v_proj")
        elif key.lower() == "o":
            lora_target_modules_2.append("out_proj")
        else:
            lora_target_modules_2.append(key)
    args.lora_target_modules_2 = lora_target_modules_2

    return args


@torch.inference_mode()
def log_validation(
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    unet,
    vae,
    args,
    accelerator,
    weight_dtype,
    global_step,
    ti_reference_mode: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompts}."
    )
    text_encoder.eval()
    text_encoder_2.eval()
    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
    unwrapped_text_encoder_2 = accelerator.unwrap_model(text_encoder_2)
    unwrapped_unet = accelerator.unwrap_model(unet)
    pipeline_cls = DiffusionPipeline if ti_reference_mode else TextBoostSDXLPipeline
    pipeline = pipeline_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        # Avoid deepcopy here: the CLIP LoRA forward methods are monkey-patched
        # with closures, and deepcopy can break those references.
        text_encoder=unwrapped_text_encoder,
        text_encoder_2=unwrapped_text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unwrapped_unet,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
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
        image = pipeline(
            **pipeline_args, num_inference_steps=20, generator=generator
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

    text_encoder.train()
    text_encoder_2.train()
    return images


def _concept_identifier_to_text(concept_identifier) -> str:
    if isinstance(concept_identifier, (list, tuple)):
        return " ".join(str(token) for token in concept_identifier if token is not None)
    return str(concept_identifier)


def _format_prompt_template(template: str, identifier_text: str) -> str:
    if "{}" in template:
        return template.format(identifier_text)
    if "<*>" in template:
        return template.replace("<*>", identifier_text)
    return template


def _log_init_token_mapping(
    tokenizer,
    encoder_name: str,
    initializer_token: str | None,
    placeholder_token_ids: list[int],
) -> None:
    placeholder_tokens = tokenizer.convert_ids_to_tokens(placeholder_token_ids)

    if initializer_token is None:
        initializer_token_ids = [None] * len(placeholder_token_ids)
        initializer_tokens = ["<none>"] * len(placeholder_token_ids)
    else:
        initializer_token_ids = tokenizer.encode(
            initializer_token,
            add_special_tokens=False,
        )
        initializer_tokens = tokenizer.convert_ids_to_tokens(initializer_token_ids)

    mapping = []
    for idx, placeholder_id in enumerate(placeholder_token_ids):
        placeholder_tok = placeholder_tokens[idx]
        if idx < len(initializer_token_ids):
            init_id = initializer_token_ids[idx]
            init_tok = initializer_tokens[idx]
        else:
            init_id = None
            init_tok = "<missing>"
        mapping.append(f"{placeholder_tok}({placeholder_id}) <- {init_tok}({init_id})")

    logger.info(
        "[%s] initializer_token=%r, num_vectors=%d, mapping: %s",
        encoder_name,
        initializer_token,
        len(placeholder_token_ids),
        ", ".join(mapping),
    )


@torch.inference_mode()
def collect_pooled_embeddings(
    text_encoder_2,
    tokenizer_2,
    prompts: list[str],
    device: torch.device,
    batch_size: int = 32,
    with_adapter: bool = True,
) -> torch.Tensor:
    if not prompts:
        return torch.empty((0, 0), dtype=torch.float32)

    was_training = text_encoder_2.training
    text_encoder_2.eval()

    pooled_chunks = []
    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i : i + batch_size]
        text_inputs = tokenizer_2(
            prompt_batch,
            truncation=True,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        if with_adapter:
            adapter_mask = None
        else:
            adapter_mask = torch.zeros_like(attention_mask)
        encoder_output_2 = text_encoder_2(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            adapter_mask=adapter_mask,
        )
        pooled_chunks.append(encoder_output_2[0].detach().float().cpu())

    if was_training:
        text_encoder_2.train()

    return torch.cat(pooled_chunks, dim=0)


def _initialize_pooled_tracker(output_dir: str, prompts: list[str]) -> dict:
    tracker_dir = Path(output_dir) / "pooled_embeddings"
    tracker_dir.mkdir(parents=True, exist_ok=True)

    prompts_file = tracker_dir / "prompts.json"
    with open(prompts_file, "w") as f:
        json.dump(prompts, f, indent=2)

    csv_file = tracker_dir / "pooled_embedding_drift.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "prompt_idx",
                "prompt",
                "pooled_norm_token_plus_adapter",
                "pooled_norm_token_only",
                "adapter_effect_norm",
                "adapter_effect_l2",
                "cos_token_plus_vs_token_only",
            ]
        )

    return {
        "dir": tracker_dir,
        "csv_file": csv_file,
        "prompts": prompts,
    }


def _log_pooled_tracker_step(
    tracker: dict,
    pooled_with_adapter: torch.Tensor,
    pooled_token_only: torch.Tensor,
    step: int,
    save_vectors: bool = True,
) -> dict[str, float]:
    pooled_a = pooled_with_adapter.detach().float().cpu()
    pooled_t = pooled_token_only.detach().float().cpu()
    adapter_effect = pooled_a - pooled_t

    pooled_norm_a = torch.norm(pooled_a, dim=-1)
    pooled_norm_t = torch.norm(pooled_t, dim=-1)
    adapter_effect_norm = torch.norm(adapter_effect, dim=-1)
    adapter_effect_l2 = torch.norm(pooled_a - pooled_t, dim=-1)
    cos_pair = F.cosine_similarity(pooled_a, pooled_t, dim=-1)

    with open(tracker["csv_file"], "a", newline="") as f:
        writer = csv.writer(f)
        for i, prompt in enumerate(tracker["prompts"]):
            writer.writerow(
                [
                    step,
                    i,
                    prompt,
                    float(pooled_norm_a[i].item()),
                    float(pooled_norm_t[i].item()),
                    float(adapter_effect_norm[i].item()),
                    float(adapter_effect_l2[i].item()),
                    float(cos_pair[i].item()),
                ]
            )

    if save_vectors:
        payload = {
            "step": step,
            "prompts": tracker["prompts"],
            "pooled_token_plus_adapter": pooled_a,
            "pooled_token_only": pooled_t,
            "pooled_adapter_effect": adapter_effect,
        }
        torch.save(payload, tracker["dir"] / f"pooled_step_{step:06d}.pt")

    return {
        "pooled_norm_token_plus_adapter_mean": float(pooled_norm_a.mean().item()),
        "pooled_norm_token_only_mean": float(pooled_norm_t.mean().item()),
        "pooled_adapter_effect_l2_mean": float(adapter_effect_l2.mean().item()),
        "pooled_cos_token_plus_vs_token_only_mean": float(cos_pair.mean().item()),
    }


def determine_scheduler_type(pretrained_model_name_or_path, revision):
    model_index_filename = "model_index.json"
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path, model_index_filename)
    else:
        model_index = hf_hub_download(
            repo_id=pretrained_model_name_or_path,
            filename=model_index_filename,
            revision=revision,
        )

    with open(model_index, "r") as f:
        scheduler_type = json.load(f)["scheduler"][1]
    return scheduler_type


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
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )

    # Load scheduler and models.
    scheduler_type = determine_scheduler_type(
        args.pretrained_model_name_or_path, args.revision
    )
    if "EDM" in scheduler_type:
        args.do_edm_style_training = True
        noise_scheduler = EDMEulerScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        logger.info("Performing EDM-style training!")
    elif args.do_edm_style_training:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        logger.info("Performing EDM-style training!")
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
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
    new_tokens = []
    added_token_ids = []
    new_token = add_new_token(
        tokenizer,
        text_encoder,
        args.placeholder_token,
        init_token=args.initializer_token,
    )
    added_token_ids += new_token.token_ids
    args.num_vectors = new_token.num_vectors
    if args.identifier_style == "custom":
        new_token.concept_identifier = [new_token.identifier, args.class_token]
    else:
        new_token.concept_identifier = new_token.identifier
    new_tokens.append(new_token)
    _log_init_token_mapping(
        tokenizer=tokenizer,
        encoder_name="text_encoder",
        initializer_token=args.initializer_token,
        placeholder_token_ids=new_token.token_ids,
    )

    new_tokens_2 = []
    added_token_ids_2 = []
    new_token_2 = add_new_token(
        tokenizer_2,
        text_encoder_2,
        args.placeholder_token,
        args.initializer_token,
    )
    added_token_ids_2 += new_token_2.token_ids
    if args.identifier_style == "custom":
        new_token_2.concept_identifier = [new_token_2.identifier, args.class_token]
    else:
        new_token_2.concept_identifier = new_token_2.identifier
    new_tokens_2.append(new_token_2)
    _log_init_token_mapping(
        tokenizer=tokenizer_2,
        encoder_name="text_encoder_2",
        initializer_token=args.initializer_token,
        placeholder_token_ids=new_token_2.token_ids,
    )

    unet.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    enable_text_adapter_1 = args.lora_rank > 0 and len(args.lora_target_modules) > 0
    enable_text_adapter_2 = args.lora_rank > 0 and len(args.lora_target_modules_2) > 0
    generator_finetune = resolve_generator_finetune_mode(args)
    generator_lora_rank = resolve_generator_lora_rank(args)
    use_unet_lora = generator_finetune == "lora"
    use_unet_kv = generator_finetune == "kv"
    use_unet_full = generator_finetune == "full"
    enable_unet_adapter = use_unet_lora or use_unet_kv or args.expand
    has_generator_training = (
        use_unet_lora or use_unet_kv or use_unet_full or args.expand
    )
    logger.info(
        "Generator fine-tuning mode: %s (lora_rank=%s)",
        generator_finetune,
        generator_lora_rank,
    )
    use_expand_bank = args.expand and args.expand_backend == "text_encoder_bank"
    if use_expand_bank and args.lora_rank <= 0:
        raise ValueError(
            "--expand_backend=text_encoder_bank requires --lora_rank > 0 "
            "for expand adapter rank."
        )
    expand_bank: ExpandAdapterBank | None = None
    ti_reference_mode = not (
        enable_text_adapter_1 or enable_text_adapter_2 or has_generator_training
    )
    # CLIP hidden state for SDXL conditioning should use penultimate layer.
    logger.info(
        "TI reference mode: %s (text_adapter_1=%s, text_adapter_2=%s, generator_training=%s)",
        ti_reference_mode,
        enable_text_adapter_1,
        enable_text_adapter_2,
        has_generator_training,
    )
    # Add LoRA.
    if enable_text_adapter_1:
        final_text_encoder_layer = text_encoder.config.num_hidden_layers - 1
        exclude_modules = [
            f"layers.{final_text_encoder_layer}.self_attn.q_proj",
            f"layers.{final_text_encoder_layer}.self_attn.k_proj",
            f"layers.{final_text_encoder_layer}.self_attn.v_proj",
            f"layers.{final_text_encoder_layer}.self_attn.out_proj",
            f"layers.{final_text_encoder_layer}.mlp.fc1",
            f"layers.{final_text_encoder_layer}.mlp.fc2",
        ]
        text_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=args.lora_target_modules,
            exclude_modules=exclude_modules,
            init_lora_weights=True,
        )
        text_encoder.add_adapter(text_lora_config)
        text_encoder.set_adapter_mask()
        text_encoder.replace_lora_forward(verbose=False)
        logger.info("Added Adapter to text_encoder")

    if enable_text_adapter_2:
        final_text_encoder_2_layer = text_encoder_2.config.num_hidden_layers - 1
        exclude_modules = [
            f"layers.{final_text_encoder_2_layer}.self_attn.q_proj",
            f"layers.{final_text_encoder_2_layer}.self_attn.k_proj",
            f"layers.{final_text_encoder_2_layer}.self_attn.v_proj",
            f"layers.{final_text_encoder_2_layer}.self_attn.out_proj",
            f"layers.{final_text_encoder_2_layer}.mlp.fc1",
            f"layers.{final_text_encoder_2_layer}.mlp.fc2",
        ]
        text_lora_config_2 = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=args.lora_target_modules_2,
            exclude_modules=exclude_modules,
            init_lora_weights=True,
        )
        text_encoder_2.add_adapter(text_lora_config_2)
        text_encoder_2.set_adapter_mask()
        text_encoder_2.replace_lora_forward(verbose=False)
        logger.info("Added Adapter to text_encoder_2")

    unet, unet_generator_params = generator_build(
        unet,
        generator_finetune,
        lora_rank=generator_lora_rank,
        lora_target_modules=["attn2.to_k", "attn2.to_v"],
        kv_patterns=[".attn2.to_k.", ".attn2.to_v."],
    )

    expand = False
    if args.expand:
        if use_expand_bank:
            target_layers = list(iter_cross_attention_to_k_layers(unet))
            expand_keys = [
                make_expand_adapter_key(name) for _, name, _ in target_layers
            ]
            expand_dims = [int(module.in_features) for _, _, module in target_layers]
            expand_bank = text_encoder_2.create_adapters(
                num_layers=len(target_layers),
                rank=args.lora_rank,
                keys=expand_keys,
                input_dims=expand_dims,
                bias=False,
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
                    attn_to_ks.append(module)
            for m in attn_to_ks:
                adapter = Adapter(2048, args.lora_rank, bias=False)
                setattr(m, "adapter", adapter)
                expand = True
            logger.info("Added Adapter to U-Net")

    if enable_unet_adapter:
        unet.set_attn_processor(build_textboost_attn_processors(unet))
    text_encoder.get_input_embeddings().requires_grad_(True)
    text_encoder_2.get_input_embeddings().requires_grad_(True)
    expand_trainable_params: list[torch.nn.Parameter] = []
    if use_expand_bank and expand_bank is not None:
        expand_trainable_params = [
            p for p in expand_bank.parameters() if p.requires_grad
        ]
    expand_param_ids = {id(p) for p in expand_trainable_params}
    num_expand_bank_params = sum(p.numel() for p in expand_trainable_params)

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
    num_text_params_2 = 0
    num_lora_params_2 = 0
    num_token_params_2 = 0
    for name, param in text_encoder_2.named_parameters():
        if "lora" in name:
            num_lora_params_2 += param.numel()
        elif "token_embedding" in name:
            num_token_params_2 += param.numel()
        elif param.requires_grad:
            num_text_params_2 += param.numel()
    num_unet_params = 0
    adapter_params = 0
    for name, param in unet.named_parameters():
        if id(param) in expand_param_ids:
            continue
        if "lora" in name:
            num_unet_params += param.numel()
        elif "adapter" in name:
            adapter_params += param.numel()
    logger.info(f"Total number of token parameters: {num_token_params:,}")
    logger.info(f"Total number of token_2 parameters: {num_token_params_2:,}")
    logger.info(
        f"Total number of trainable text_encoder parameters: {num_text_params:,}"
    )
    logger.info(
        f"Total number of trainable text_encoder_2 parameters: {num_text_params_2:,}"
    )
    logger.info(f"Total number of text_encoder LoRA parameters: {num_lora_params:,}")
    logger.info(
        f"Total number of text_encoder_2 LoRA parameters: {num_lora_params_2:,}"
    )
    logger.info(
        "Total number of text-side expand adapter parameters: %s",
        f"{num_expand_bank_params:,}",
    )
    logger.info(f"Total number of Adapter parameters: {adapter_params:,}")
    logger.info(f"Total number of U-Net LoRA parameters: {num_unet_params:,}")

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
                    # text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(
                    #     get_peft_model_state_dict(model)
                    # )
                    (
                        unwrap_model(model).save_pretrained(
                            str(adapter_artifact_dir(output_dir, "text_encoder"))
                        )
                    )
                elif isinstance(model, type(unwrap_model(text_encoder_2))):
                    (
                        unwrap_model(model).save_pretrained(
                            str(adapter_artifact_dir(output_dir, "text_encoder_2"))
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
            text_encoder_adapter_dir = adapter_artifact_dir(
                output_dir, "text_encoder_2"
            )
            active_expand_bank = unwrap_model(text_encoder_2).get_expand_adapter_bank()
            if active_expand_bank is None:
                raise ValueError(
                    "Expand backend is text_encoder_bank, but no bank was found on text encoder 2."
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
    emb_optimizer = torch.optim.AdamW(
        (
            list(text_encoder.get_input_embeddings().parameters())
            + list(text_encoder_2.get_input_embeddings().parameters())
        ),
        lr=args.emb_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # Print number of parameters to optimizer.
    params_to_optimize = [
        {
            "params": list(
                filter(
                    lambda p: p.requires_grad,
                    text_encoder.text_model.encoder.parameters(),
                )
            ),
        },
        {
            "params": list(
                filter(
                    lambda p: p.requires_grad,
                    text_encoder_2.text_model.encoder.parameters(),
                )
            ),
        },
    ]
    if expand_trainable_params:
        params_to_optimize.append(
            {
                "params": expand_trainable_params,
                "lr": args.adapter_learning_rate,
            }
        )
    elif args.expand:
        params_to_optimize.append(
            {
                "params": list(filter(lambda p: p.requires_grad, unet.parameters())),
                "lr": args.adapter_learning_rate,
            }
        )
    if use_unet_lora or use_unet_kv or use_unet_full:
        unet_lora_trainable_params = [
            p for p in unet_generator_params if id(p) not in expand_param_ids
        ]
        known_param_ids = {id(p) for p in unet_lora_trainable_params}
        unet_lora_trainable_params += [
            p
            for p in unet.parameters()
            if p.requires_grad
            and id(p) not in expand_param_ids
            and id(p) not in known_param_ids
        ]
        params_to_optimize.append(
            {
                "params": unet_lora_trainable_params,
                "lr": args.unet_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    with torch.no_grad():
        total_params = 0
        for group in params_to_optimize:
            total_params += sum(p.numel() for p in group["params"])
        logger.info(f"Total number of parameters to optimize: {total_params:,}")

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
    train_dataset_raw = SDXLDataset(
        data_path=args.data_dir,
        concept_identifier=new_token.concept_identifier,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        num_instance=args.num_samples,
        template=args.template,
        size=args.resolution,
        center_crop=args.center_crop,
        hflip=False if ti_reference_mode else True,
    )
    train_dataset = (
        train_dataset_raw.with_drop_last(False).shuffle(seed=args.seed).repeat()
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    text_encoder.train()
    text_encoder_2.train()
    # Prepare everything with our `accelerator`.
    if use_unet_lora or use_unet_kv or use_unet_full or use_expand_bank:
        (
            text_encoder,
            text_encoder_2,
            unet,
            emb_optimizer,
            optimizer,
            emb_lr_scheduler,
            lr_scheduler,
        ) = accelerator.prepare(
            text_encoder,
            text_encoder_2,
            unet,
            emb_optimizer,
            optimizer,
            emb_lr_scheduler,
            lr_scheduler,
        )
    else:
        (
            text_encoder,
            text_encoder_2,
            emb_optimizer,
            optimizer,
            emb_lr_scheduler,
            lr_scheduler,
        ) = accelerator.prepare(
            text_encoder,
            text_encoder_2,
            emb_optimizer,
            optimizer,
            emb_lr_scheduler,
            lr_scheduler,
        )

    active_expand_bank: ExpandAdapterBank | None = None
    if use_expand_bank and expand:
        active_expand_bank = unwrap_model(text_encoder_2).get_expand_adapter_bank()
        if active_expand_bank is None:
            raise ValueError(
                "Expand backend is text_encoder_bank, but no bank was found on text encoder 2."
            )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder_2 to device and cast to weight_dtype.
    unet.to(accelerator.device, dtype=weight_dtype)
    vae_dtype = (
        torch.float32
        if (use_unet_lora or use_unet_kv or use_unet_full)
        else weight_dtype
    )
    vae.to(accelerator.device, dtype=vae_dtype)
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # accelerator.init_trackers("textboost-sdxl", config=vars(args))

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

    pooled_log_steps = (
        args.pooled_embedding_log_steps
        if args.pooled_embedding_log_steps > 0
        else args.validation_steps
    )

    def save_checkpoint_artifacts(save_path: str, checkpoint_step: int) -> None:
        save_unet_custom_diffusion_artifact(save_path)
        save_unet_full_artifact(save_path)
        if expand:
            save_expand_artifact(save_path)
        save_embeddings(
            accelerator.unwrap_model(text_encoder),
            new_tokens,
            os.path.join(save_path, "learned_embeds.bin"),
            safe_serialization=False,
        )
        save_embeddings(
            accelerator.unwrap_model(text_encoder_2),
            new_tokens_2,
            os.path.join(save_path, "learned_embeds_2.bin"),
            safe_serialization=False,
        )
        write_checkpoint_metadata(
            save_path,
            step=checkpoint_step,
            args=args,
            has_text_encoder_adapter=args.lora_rank > 0
            and bool(args.lora_target_modules),
            has_text_encoder_2_adapter=args.lora_rank > 0
            and bool(args.lora_target_modules_2),
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
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            unet,
            vae,
            args,
            accelerator,
            weight_dtype,
            validation_step,
            ti_reference_mode=ti_reference_mode,
        )

    def initialize_pooled_tracker() -> dict:
        prompt_templates = (
            args.pooled_embedding_prompts
            if args.pooled_embedding_prompts is not None
            else list(train_dataset_raw.template)
        )
        identifier_text = _concept_identifier_to_text(new_token_2.concept_identifier)
        pooled_prompts = [
            _format_prompt_template(template, identifier_text)
            for template in prompt_templates
        ]
        pooled_prompts = list(dict.fromkeys(pooled_prompts))
        tracker = _initialize_pooled_tracker(args.output_dir, pooled_prompts)
        logger.info(
            "Initialized pooled embedding tracker with %s prompts. log_steps=%s, save_vectors=%s",
            len(pooled_prompts),
            pooled_log_steps,
            args.save_pooled_embedding_vectors,
        )
        return tracker

    def collect_pooled_with_adapter(tracker: dict):
        return collect_pooled_embeddings(
            accelerator.unwrap_model(text_encoder_2),
            tokenizer_2,
            tracker["prompts"],
            accelerator.device,
            batch_size=args.pooled_embedding_batch_size,
            with_adapter=True,
        )

    def collect_pooled_token_only(tracker: dict):
        return collect_pooled_embeddings(
            accelerator.unwrap_model(text_encoder_2),
            tokenizer_2,
            tracker["prompts"],
            accelerator.device,
            batch_size=args.pooled_embedding_batch_size,
            with_adapter=False,
        )

    def summarize_pooled(
        tracker: dict,
        pooled_with_adapter,
        pooled_token_only,
        pool_step: int,
    ) -> dict[str, float]:
        return _log_pooled_tracker_step(
            tracker,
            pooled_with_adapter,
            pooled_token_only,
            step=pool_step,
            save_vectors=args.save_pooled_embedding_vectors,
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
            PooledEmbeddingTrackerCallback(
                accelerator=accelerator,
                enabled=args.track_pooled_embeddings,
                log_steps=pooled_log_steps,
                initialize_fn=initialize_pooled_tracker,
                collect_with_adapter_fn=collect_pooled_with_adapter,
                collect_token_only_fn=collect_pooled_token_only,
                summarize_fn=summarize_pooled,
            ),
        ]
    )

    accelerator.wait_for_everyone()

    text_encoder.train()

    # keep original embeddings as reference
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight.data.clone()
    )
    orig_embeds_params_2 = (
        accelerator.unwrap_model(text_encoder_2)
        .get_input_embeddings()
        .weight.data.clone()
    )

    start_time = time.perf_counter()
    trainer = SDXLTrainer(
        accelerator=accelerator,
        train_dataloader=train_dataloader,
        max_train_steps=args.max_train_steps,
        progress_bar=progress_bar,
        callbacks=callbacks,
        unet=unet,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        noise_scheduler=noise_scheduler,
        emb_optimizer=emb_optimizer,
        optimizer=optimizer,
        emb_lr_scheduler=emb_lr_scheduler,
        lr_scheduler=lr_scheduler,
        args=args,
        weight_dtype=weight_dtype,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        added_token_ids=added_token_ids,
        added_token_ids_2=added_token_ids_2,
        orig_embeds_params=orig_embeds_params,
        orig_embeds_params_2=orig_embeds_params_2,
        enable_unet_adapter=enable_unet_adapter,
        active_expand_bank=active_expand_bank,
        expand_param_ids=expand_param_ids,
        ti_reference_mode=ti_reference_mode,
        accumulate_models=(unet, text_encoder, text_encoder_2),
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

        if args.lora_rank > 0 and args.lora_target_modules:
            text_encoder = unwrap_model(text_encoder).to(torch.float32)
            text_encoder.save_pretrained(
                str(adapter_artifact_dir(args.output_dir, "text_encoder"))
            )
        if args.lora_rank > 0 and args.lora_target_modules_2:
            text_encoder_2 = unwrap_model(text_encoder_2).to(torch.float32)
            text_encoder_2.save_pretrained(
                str(adapter_artifact_dir(args.output_dir, "text_encoder_2"))
            )

        if expand:
            save_expand_artifact(args.output_dir)

        save_embeddings(
            accelerator.unwrap_model(text_encoder),
            new_tokens,
            os.path.join(args.output_dir, "learned_embeds.bin"),
            safe_serialization=False,
        )
        save_embeddings(
            accelerator.unwrap_model(text_encoder_2),
            new_tokens_2,
            os.path.join(args.output_dir, "learned_embeds_2.bin"),
            safe_serialization=False,
        )
        write_checkpoint_metadata(
            args.output_dir,
            step=args.max_train_steps,
            args=args,
            has_text_encoder_adapter=args.lora_rank > 0
            and bool(args.lora_target_modules),
            has_text_encoder_2_adapter=args.lora_rank > 0
            and bool(args.lora_target_modules_2),
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
