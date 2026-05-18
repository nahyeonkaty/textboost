#!/usr/bin/env python3
import argparse
import copy
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import safetensors
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm
from transformers import AutoTokenizer

import diffusers
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaTransformer2DModel,
    SanaPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    is_wandb_available,
    make_image_grid,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from textboost.datasets import SanaDataset
from textboost.ti_utils import add_new_token, forced_weight_norm
from textboost.text_encoders.gemma2 import TextModel

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


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
        default=None,
        help="Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.",
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
        default=1e-4,
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
        # default=["to_k", "to_q", "to_v"],
        default=["to_k", "to_q", "to_v", "to_out.0"],
        help="Target modules for LoRA.",
    )

    parser.add_argument(
        "--max_embedding_norm",
        type=float,
        default=2.5,  # heuristically chosen.
        help="Max norm for the embedding.",
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
        choices=["custom", "ti"],
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
            lora_target_modules.append("o_proj")
        else:
            lora_target_modules.append(key)
    args.lora_target_modules = lora_target_modules

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
    pipeline = SanaPipeline.from_pretrained(
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
    identifier = "".join(identifier)
    for validation_prompt in args.validation_prompts:
        pipeline_args = {
            "prompt": validation_prompt.replace("<*>", identifier),
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


def save_embeddings(
    text_encoder, new_tokens, accelerator, save_path, safe_serialization=True
):
    logger.info("Saving embeddings")
    learned_embeds_dict = {}
    for key, value in new_tokens.items():
        token_ids = value["token_id"]
        learned_embeds = (
            accelerator.unwrap_model(text_encoder)
            .get_input_embeddings()
            .weight[min(token_ids) : max(token_ids) + 1]
        )
        learned_embeds_dict[key] = learned_embeds.detach().cpu()

    if safe_serialization:
        safetensors.torch.save_file(
            learned_embeds_dict, save_path, metadata={"format": "pt"}
        )
    else:
        torch.save(learned_embeds_dict, save_path)


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
    text_encoder = TextModel.from_pretrained(
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
    new_tokens = {}
    placeholder_token_ids = []

    new_token = add_new_token(
        tokenizer,
        text_encoder,
        args.placeholder_token,
        args.initializer_token,
    )
    placeholder_token_ids += new_token.token_ids
    args.num_vectors = new_token.num_vectors
    new_tokens[args.placeholder_token] = {
        "token_id": new_token.token_ids,
        "placeholder": new_token.identifier,
        "num_vectors": new_token.num_vectors,
        # "identifier": [identifier, args.class_token],
    }
    if args.identifier_style == "ti":
        new_tokens[args.placeholder_token]["identifier"] = identifier
    else:
        new_tokens[args.placeholder_token]["identifier"] = [
            identifier,
            args.class_token,
        ]

    # print(tokenizer)
    print(placeholder_token_ids)
    print(new_tokens)

    # We only train the additional adapter LoRA layers
    dit.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder.get_input_embeddings().requires_grad_(True)

    # Add LoRA.
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=args.lora_target_modules,
    )
    dit.add_adapter(lora_config)

    num_token_params = 0
    num_te_params = 0
    num_te_adapter_params = 0
    for name, param in text_encoder.named_parameters():
        if "lora" in name:
            num_te_adapter_params += param.numel()
        elif "embed_tokens" in name:
            num_token_params += param.numel()
        elif param.requires_grad:
            num_te_params += param.numel()
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
    logger.info(
        f"Total number of trainable DiT parameters: {num_unet_adapter_params:,}"
    )
    logger.info(f"Total number of DiT Adapter parameters: {num_dit_params:,}")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            dit_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(dit))):
                    dit_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, type(unwrap_model(text_encoder))):
                    pass
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            if dit_lora_layers_to_save is not None:
                SanaPipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=dit_lora_layers_to_save,
                )

    accelerator.register_save_state_pre_hook(save_model_hook)

    text_encoding_pipeline = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=None,
        transformer=None,
    )

    if args.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()

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
    params_to_optimize = [
        {
            "params": list(filter(lambda p: p.requires_grad, dit.parameters())),
        }
    ]
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
    hflip = "true"
    train_dataset = SanaDataset(
        data_path=args.data_dir,
        concept_identifier=new_tokens[args.placeholder_token]["identifier"],
        num_instance=args.num_samples,
        template=args.template,
        class_token=args.class_token,
        size=args.resolution,
        center_crop=args.center_crop,
        hflip=hflip,
    )
    train_dataset = train_dataset.with_drop_last(False).shuffle(seed=args.seed).repeat()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
        # prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
        #     prompt,
        #     max_sequence_length=args.max_sequence_length,
        #     complex_human_instruction=args.complex_human_instruction,
        # )
        prompt_embeds, prompt_attention_mask = (
            text_encoding_pipeline._get_gemma_prompt_embeds(
                prompt,
                device=accelerator.device,
                dtype=unwrap_model(dit).dtype,
                max_sequence_length=args.max_sequence_length,
                complex_human_instruction=args.complex_human_instruction,
            )
        )
        # if args.offload:
        #     text_encoding_pipeline = text_encoding_pipeline.to("cpu")
        return prompt_embeds, prompt_attention_mask

    # Prepare everything with our `accelerator`.
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
        dit.to(accelerator.device, dtype=torch.float16)
    else:
        dit.to(accelerator.device, dtype=torch.bfloat16)
    for name, module in dit.named_modules():
        if "adapter" in name:
            module.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        # accelerator.init_trackers("textboost", config=tracker_config)
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

    train_iterator = iter(train_dataloader)

    # keep original embeddings as reference
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight.data.clone()
    )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    start_time = time.perf_counter()
    while step < args.max_train_steps:
        batch = next(train_iterator)
        prompt = batch["prompt"]

        pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)

        # Convert images to latent space
        model_input = vae.encode(pixel_values).latent
        model_input = model_input * vae.config.scaling_factor
        model_input = model_input.to(dtype=weight_dtype)

        with accelerator.accumulate(dit, text_encoder):
            # Sample noise that we'll add to the model input.
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(
                device=model_input.device
            )

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(
                timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
            )
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            # Get the text embedding for conditioning.
            prompt_embeds, prompt_attention_mask = compute_text_embeddings(
                prompt,
                text_encoding_pipeline,
            )

            # Predict the noise residual.
            model_pred = dit(
                hidden_states=noisy_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                encoder_attention_mask=prompt_attention_mask,
                return_dict=False,
            )[0]

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=args.weighting_scheme, sigmas=sigmas
            )

            # flow matching loss
            target = noise - model_input

            # Compute regular loss.
            loss = torch.mean(
                (
                    weighting.float() * (model_pred.float() - target.float()) ** 2
                ).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = dit.parameters()
                # for p in params_to_clip:
                #     print(p.grad)
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            emb_optimizer.step()
            optimizer.step()
            emb_lr_scheduler.step()
            lr_scheduler.step()
            emb_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)

            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[
                min(placeholder_token_ids) : max(placeholder_token_ids) + 1
            ] = False

            with torch.no_grad():
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]

            norm = forced_weight_norm(
                text_encoder=accelerator.unwrap_model(text_encoder),
                index=placeholder_token_ids,
                magnitude=args.max_embedding_norm,
            )
            accelerator.log({"v_norm": norm.mean()}, step=step)
            # print(norm)

        # Checks if the accelerator has performed an optimization step behind the scenes.
        if accelerator.sync_gradients:
            progress_bar.update(1)
            step += 1

            if accelerator.is_main_process:
                if step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (
                                len(checkpoints) - args.checkpoints_total_limit + 1
                            )
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint
                                )
                                shutil.rmtree(removing_checkpoint)
                    save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                    accelerator.save_state(
                        save_path
                    )  # NOTE: Requires too much storage space.
                    logger.info(f"Saved state to {save_path}")

                    # Save the embeddings.
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    save_embeddings(
                        text_encoder,
                        new_tokens,
                        accelerator,
                        os.path.join(ckpt_dir, "learned_embeds.bin"),
                        safe_serialization=False,
                    )

                images = []

                if args.validation_prompts and step % args.validation_steps == 0:
                    images = log_validation(
                        tokenizer,
                        text_encoder,
                        vae,
                        dit,
                        args,
                        accelerator,
                        step,
                    )
                    if images:
                        rows = len(args.validation_prompts)
                        cols = args.num_validation_images
                        image_grid = make_image_grid(images, rows, cols)
                        image_grid.save(
                            os.path.join(args.output_dir, f"validation_{step}.jpg")
                        )

        logs = {
            "loss": loss.detach().item(),
            "lr_emb": emb_lr_scheduler.get_last_lr()[0],
            "lr": lr_scheduler.get_last_lr()[0],
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.lora_rank > 0:
            dit = unwrap_model(dit).to(torch.float32)
            dit.save_pretrained(os.path.join(args.output_dir, "transformer"))

        save_embeddings(
            text_encoder,
            new_tokens,
            accelerator,
            os.path.join(args.output_dir, "learned_embeds.bin"),
            safe_serialization=False,
        )

    end_time = time.perf_counter()
    logger.info(f"Training took {end_time - start_time:.2f} seconds")
    accelerator.end_training()


if __name__ == "__main__":
    main()
