from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import (
    SanaTransformer2DModel,
    UNet2DConditionModel,
)

from textboost.attention_processor import (
    build_textboost_attn_processors,
)
from textboost.pipelines.sd import TextBoostPipeline
from textboost.pipelines.sdxl import TextBoostSDXLPipeline
from textboost.pipelines.sana import TextBoostSanaPipeline
from textboost.adapters import Adapter, attach_adapters_to_model, TrfConfig
from textboost.expand_bank import (
    build_expand_bank_from_state_dict,
    iter_cross_attention_to_k_layers,
    make_expand_adapter_key,
)
from textboost.text_encoders.clip import TextModel as ClipTextModel
from textboost.text_encoders.clip import TextModelWithProjection
from textboost.text_encoders.gemma2 import TextModel as Gemma2TextModel
from textboost.ti_utils import load_new_token

UNET_CUSTOM_DIFFUSION_KV_FILENAME = "custom_diffusion_kv.bin"
TRANSFORMER_CUSTOM_DIFFUSION_KV_FILENAME = "custom_diffusion_kv.bin"


def _is_adapter_checkpoint_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    names = {p.name for p in path.iterdir()}
    if "adapter_config.json" in names:
        return True
    return any(
        name in {"adapter_model.bin", "adapter_model.safetensors"} for name in names
    )


def _resolve_text_adapter_dir(checkpoint_path: Path, module_name: str) -> Path | None:
    candidates = [
        checkpoint_path / "adapter" / module_name,
        checkpoint_path / module_name,
    ]
    for candidate in candidates:
        if _is_adapter_checkpoint_dir(candidate):
            return candidate
    return None


def _resolve_unet_lora_file(checkpoint_path: Path) -> Path | None:
    candidates = [
        checkpoint_path / "adapter" / "unet" / "pytorch_lora_weights.safetensors",
        checkpoint_path / "pytorch_lora_weights.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _is_diffusers_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    weight_files = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    )
    return any((path / filename).exists() for filename in weight_files)


def _resolve_unet_full_dir(checkpoint_path: Path) -> Path | None:
    candidates = [
        checkpoint_path / "adapter" / "unet",
        checkpoint_path / "unet",
    ]
    for candidate in candidates:
        if _is_diffusers_model_dir(candidate):
            return candidate
    return None


def _resolve_unet_expand_adapter_file(checkpoint_path: Path) -> Path | None:
    candidates = [
        checkpoint_path / "adapter" / "unet" / "adapter.bin",
        checkpoint_path / "adapter.bin",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_unet_custom_diffusion_kv_file(checkpoint_path: Path) -> Path | None:
    candidates = [
        checkpoint_path / "adapter" / "unet" / UNET_CUSTOM_DIFFUSION_KV_FILENAME,
        checkpoint_path / UNET_CUSTOM_DIFFUSION_KV_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_transformer_lora_file(checkpoint_path: Path) -> Path | None:
    candidates = [
        checkpoint_path
        / "adapter"
        / "transformer"
        / "pytorch_lora_weights.safetensors",
        checkpoint_path / "adapter" / "unet" / "pytorch_lora_weights.safetensors",
        checkpoint_path / "pytorch_lora_weights.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_transformer_custom_diffusion_kv_file(checkpoint_path: Path) -> Path | None:
    candidates = [
        checkpoint_path
        / "adapter"
        / "transformer"
        / TRANSFORMER_CUSTOM_DIFFUSION_KV_FILENAME,
        checkpoint_path / "adapter" / "unet" / TRANSFORMER_CUSTOM_DIFFUSION_KV_FILENAME,
        checkpoint_path / TRANSFORMER_CUSTOM_DIFFUSION_KV_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_transformer_full_dir(checkpoint_path: Path) -> Path | None:
    candidates = [
        checkpoint_path / "adapter" / "transformer",
        checkpoint_path / "adapter" / "unet",
        checkpoint_path / "transformer",
    ]
    for candidate in candidates:
        if _is_diffusers_model_dir(candidate):
            return candidate
    return None


def _resolve_unet_expand_bank_file(checkpoint_path: Path) -> Path | None:
    candidates = [checkpoint_path / "adapter" / "unet" / "expand_bank.bin"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_expand_bank_file(checkpoint_path: Path, module_name: str) -> Path | None:
    legacy_module_name = (
        "text_encoder_2" if module_name == "text_encoder" else "text_encoder"
    )
    candidates = [
        checkpoint_path / "adapter" / module_name / "expand_bank.bin",
        checkpoint_path / module_name / "expand_bank.bin",
        checkpoint_path / "adapter" / legacy_module_name / "expand_bank.bin",
        checkpoint_path / legacy_module_name / "expand_bank.bin",
        checkpoint_path / "expand_bank.bin",
    ]
    legacy_unet_path = _resolve_unet_expand_bank_file(checkpoint_path)
    if legacy_unet_path is not None:
        candidates.append(legacy_unet_path)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_sd_pipeline(
    model: str,
    checkpoint_path: str | Path,
    enable_cpa_mask: bool = True,
) -> tuple[TextBoostPipeline | TextBoostSDXLPipeline, list[str]]:
    checkpoint_path = Path(checkpoint_path)
    sdxl = "stable-diffusion-xl" in model
    print(f"SDXL: {sdxl}")

    unet_full_dir = _resolve_unet_full_dir(checkpoint_path)
    if unet_full_dir is not None:
        unet = UNet2DConditionModel.from_pretrained(str(unet_full_dir))
        print(f"Loaded full U-Net weights from checkpoint: {unet_full_dir}")
    else:
        unet = UNet2DConditionModel.from_pretrained(
            model,
            subfolder="unet",
        )

    text_encoder = ClipTextModel.from_pretrained(
        model,
        subfolder="text_encoder",
    )
    text_encoder_path = _resolve_text_adapter_dir(checkpoint_path, "text_encoder")
    if text_encoder_path is not None:
        text_encoder.load_adapter(str(text_encoder_path))
        if enable_cpa_mask:
            text_encoder.set_adapter_mask()
            text_encoder.replace_lora_forward()

    if sdxl:
        text_encoder_2 = TextModelWithProjection.from_pretrained(
            model,
            subfolder="text_encoder_2",
        )
        text_encoder_2_path = _resolve_text_adapter_dir(
            checkpoint_path, "text_encoder_2"
        )
        if text_encoder_2_path is not None:
            text_encoder_2.load_adapter(str(text_encoder_2_path))
            if enable_cpa_mask:
                text_encoder_2.set_adapter_mask()
                text_encoder_2.replace_lora_forward()
        pipeline = TextBoostSDXLPipeline.from_pretrained(
            model,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            unet=unet,
            safety_checker=None,
        )
    else:
        pipeline = TextBoostPipeline.from_pretrained(
            model,
            text_encoder=text_encoder,
            unet=unet,
            safety_checker=None,
        )

    # 2. Optionally load U-Net LoRA weights or Adapter.
    unet_lora_file = _resolve_unet_lora_file(checkpoint_path)
    unet_custom_diffusion_kv_file = _resolve_unet_custom_diffusion_kv_file(
        checkpoint_path
    )
    expand_bank_owner = text_encoder_2 if sdxl else text_encoder
    unet_expand_bank_file = _resolve_expand_bank_file(
        checkpoint_path,
        "text_encoder_2" if sdxl else "text_encoder",
    )
    unet_expand_file = _resolve_unet_expand_adapter_file(checkpoint_path)
    if unet_full_dir is not None:
        unet_lora_file = None
        unet_custom_diffusion_kv_file = None
        unet_expand_file = None
    if unet_lora_file is not None:
        # U-Net adapters in this project are trained with TextBoostAttnProcessor.
        # This is independent of text-encoder CPA masking.
        unet.set_attn_processor(build_textboost_attn_processors(unet))
        pipeline.load_lora_weights(
            str(unet_lora_file.parent),
            weight_name=unet_lora_file.name,
        )
        print("Loaded LoRA weights from checkpoint.")
    elif unet_custom_diffusion_kv_file is not None:
        unet_kv_state_dict = torch.load(
            unet_custom_diffusion_kv_file, map_location="cpu"
        )
        expected_kv_keys = {
            key
            for key in unet.state_dict().keys()
            if ".attn2.to_k." in key or ".attn2.to_v." in key
        }
        missing_kv_keys = sorted(expected_kv_keys - set(unet_kv_state_dict.keys()))
        if missing_kv_keys:
            raise ValueError(
                "Missing cross-attention K/V keys in Custom Diffusion checkpoint: "
                f"{missing_kv_keys[:5]}{'...' if len(missing_kv_keys) > 5 else ''}"
            )
        unexpected_kv_keys = sorted(
            key for key in unet_kv_state_dict.keys() if key not in expected_kv_keys
        )
        if unexpected_kv_keys:
            raise ValueError(
                "Unexpected keys in Custom Diffusion K/V checkpoint: "
                f"{unexpected_kv_keys[:5]}{'...' if len(unexpected_kv_keys) > 5 else ''}"
            )
        unet.load_state_dict(unet_kv_state_dict, strict=False)
        print("Loaded Custom Diffusion full K/V weights from checkpoint.")
    elif unet_expand_bank_file is not None:
        expand_state_dict = torch.load(unet_expand_bank_file, map_location="cpu")
        expand_bank = build_expand_bank_from_state_dict(expand_state_dict)
        matched_layers = [
            (idx, name)
            for idx, name, _module in iter_cross_attention_to_k_layers(unet)
            if make_expand_adapter_key(name) in expand_bank.adapters
        ]
        if hasattr(expand_bank_owner, "set_expand_adapter_bank"):
            expand_bank_owner.set_expand_adapter_bank(expand_bank)
        else:
            raise ValueError("Loaded text encoder does not support expand adapters.")
        unet.set_attn_processor(build_textboost_attn_processors(unet))
        print(
            f"Loaded Expand Adapter Bank from checkpoint. layers={len(matched_layers)}"
        )
    elif unet_expand_file is not None:
        # U-Net adapters in this project are trained with TextBoostAttnProcessor.
        # This is independent of text-encoder CPA masking.
        unet.set_attn_processor(build_textboost_attn_processors(unet))
        adapter_state_dict = torch.load(unet_expand_file, map_location="cpu")
        for k, v in adapter_state_dict.items():
            if "adapter.up" in k:
                dim = v.shape[0]
                rank = v.shape[1]
                bias = k.replace("weight", "bias") in adapter_state_dict
                break
        attn_to_ks = []
        attn_to_vs = []
        for name, module in unet.named_modules():
            if "attn2.to_k" in name:
                attn_to_ks.append(module)
            elif "attn2.to_v" in name:
                attn_to_vs.append(module)
        for m in attn_to_ks:
            adapter = Adapter(dim, rank, bias)
            setattr(m, "adapter", adapter)
        for m in attn_to_vs:
            adapter = Adapter(dim, rank, bias)
            setattr(m, "adapter", adapter)
        unet.load_state_dict(adapter_state_dict, strict=False)
        print("Loaded Adapter from checkpoint.")

    # 3. Load learned embeddings.
    emb_dict = torch.load(checkpoint_path / "learned_embeds.bin")
    identifiers = []
    for key, value in emb_dict.items():
        identifier = load_new_token(
            text_encoder,
            pipeline.tokenizer,
            placeholder=key,
            learned_embedding=value,
        )
        identifiers.append(identifier)
    if sdxl:
        emb_dict = torch.load(checkpoint_path / "learned_embeds_2.bin")
        for key, value in emb_dict.items():
            identifier = load_new_token(
                text_encoder_2,
                pipeline.tokenizer_2,
                placeholder=key,
                learned_embedding=value,
            )
            identifiers.append(identifier)
    print("Loaded learned embeddings from checkpoint.")
    print(identifiers)
    return pipeline, identifiers


def load_sana_pipeline(
    model_name: str, checkpoint_path: str | Path
) -> tuple[TextBoostSanaPipeline, list[str]]:
    checkpoint_path = Path(checkpoint_path)

    text_encoder = Gemma2TextModel.from_pretrained(model_name, subfolder="text_encoder")
    transformer_full_dir = _resolve_transformer_full_dir(checkpoint_path)
    if transformer_full_dir is not None:
        transformer = SanaTransformer2DModel.from_pretrained(str(transformer_full_dir))
        pipeline = TextBoostSanaPipeline.from_pretrained(
            model_name,
            text_encoder=text_encoder,
            transformer=transformer,
        )
        print(
            f"Loaded full transformer weights from checkpoint: {transformer_full_dir}"
        )
    else:
        pipeline = TextBoostSanaPipeline.from_pretrained(
            model_name,
            text_encoder=text_encoder,
        )

    config_path = None
    for candidate in (
        checkpoint_path / "config.json",
        checkpoint_path / "text_encoder" / "config.json",
        checkpoint_path / "adapter" / "text_encoder" / "config.json",
    ):
        if candidate.exists():
            config_path = candidate
            break
    if config_path is not None:
        with open(config_path, "r") as f:
            config = json.load(f)
        config = TrfConfig(**config)
        pipeline.text_encoder = attach_adapters_to_model(pipeline.text_encoder, config)

    text_encoder_path = None
    for candidate in (
        checkpoint_path / "text_encoder.bin",
        checkpoint_path / "text_encoder" / "text_encoder.bin",
        checkpoint_path / "adapter" / "text_encoder" / "text_encoder.bin",
    ):
        if candidate.exists():
            text_encoder_path = candidate
            break

    if text_encoder_path is not None:
        state_dict = torch.load(text_encoder_path, map_location="cpu")
        pipeline.text_encoder.load_state_dict(state_dict, strict=False)

    # 2. Optionally load transformer LoRA/KV weights.
    transformer_lora_file = _resolve_transformer_lora_file(checkpoint_path)
    transformer_custom_diffusion_kv_file = (
        _resolve_transformer_custom_diffusion_kv_file(checkpoint_path)
    )
    if transformer_full_dir is not None:
        transformer_lora_file = None
        transformer_custom_diffusion_kv_file = None
    if transformer_lora_file is not None:
        pipeline.load_lora_weights(
            str(transformer_lora_file.parent),
            weight_name=transformer_lora_file.name,
        )
        print("Loaded LoRA weights from checkpoint.")
    elif transformer_custom_diffusion_kv_file is not None:
        transformer_kv_state_dict = torch.load(
            transformer_custom_diffusion_kv_file, map_location="cpu"
        )
        expected_keys = set(pipeline.transformer.state_dict().keys())
        unexpected_keys = sorted(
            key for key in transformer_kv_state_dict.keys() if key not in expected_keys
        )
        if unexpected_keys:
            raise ValueError(
                "Unexpected keys in transformer K/V checkpoint: "
                f"{unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}"
            )
        pipeline.transformer.load_state_dict(transformer_kv_state_dict, strict=False)
        print("Loaded transformer Custom Diffusion K/V weights from checkpoint.")

    expand_bank_file = _resolve_expand_bank_file(checkpoint_path, "text_encoder")
    if expand_bank_file is not None:
        expand_state_dict = torch.load(expand_bank_file, map_location="cpu")
        expand_bank = build_expand_bank_from_state_dict(expand_state_dict)
        if not hasattr(pipeline.text_encoder, "set_expand_adapter_bank"):
            raise ValueError("Loaded text encoder does not support expand adapters.")
        pipeline.text_encoder.set_expand_adapter_bank(expand_bank)
        pipeline.transformer.set_attn_processor(
            build_textboost_attn_processors(pipeline.transformer, use_sana=True)
        )
        print(f"Loaded text-encoder expand bank from checkpoint: {expand_bank_file}")
    else:
        legacy_expand_adapter_file = _resolve_unet_expand_adapter_file(checkpoint_path)
        if legacy_expand_adapter_file is not None:
            adapter_state_dict = torch.load(
                legacy_expand_adapter_file, map_location="cpu"
            )
            dim = None
            rank = None
            bias = False
            for key, value in adapter_state_dict.items():
                if "adapter.up" in key:
                    dim = value.shape[0]
                    rank = value.shape[1]
                    bias = key.replace("weight", "bias") in adapter_state_dict
                    break
            if dim is None or rank is None:
                raise ValueError(
                    "Could not infer legacy SANA adapter shape from checkpoint."
                )
            for _idx, _name, module in iter_cross_attention_to_k_layers(
                pipeline.transformer
            ):
                adapter = Adapter(int(dim), int(rank), bias)
                setattr(module, "adapter", adapter)
            missing, unexpected = pipeline.transformer.load_state_dict(
                adapter_state_dict,
                strict=False,
            )
            if missing:
                print(f"Missing keys in legacy SANA adapter state dict: {missing}")
            if unexpected:
                print(
                    f"Unexpected keys in legacy SANA adapter state dict: {unexpected}"
                )
            pipeline.transformer.set_attn_processor(
                build_textboost_attn_processors(pipeline.transformer, use_sana=True)
            )
            print(
                f"Loaded legacy DiT-attached expand adapters from checkpoint: {legacy_expand_adapter_file}"
            )

    # 3. Load learned embeddings.
    emb_dict = torch.load(checkpoint_path / "learned_embeds.bin")
    identifiers = []
    for key, value in emb_dict.items():
        identifier = load_new_token(
            pipeline.text_encoder,
            pipeline.tokenizer,
            placeholder=key,
            learned_embedding=value,
            joiner="",
        )
        identifiers.append(identifier)

    print("Loaded learned embeddings from checkpoint.")
    print(identifiers)
    return pipeline, identifiers
