from __future__ import annotations

from collections.abc import Sequence

import torch
from peft import LoraConfig


def _default_lora_targets(generator: torch.nn.Module) -> list[str]:
    module_names = [name for name, _ in generator.named_modules()]
    if any("attn2.to_k" in name for name in module_names):
        return ["attn2.to_k", "attn2.to_v"]
    return ["to_q", "to_k", "to_v", "to_out.0"]


def _default_kv_patterns(generator: torch.nn.Module) -> list[str]:
    param_names = [name for name, _ in generator.named_parameters()]
    if any(".attn2.to_k." in name for name in param_names):
        return [".attn2.to_k.", ".attn2.to_v."]
    if any(".to_k." in name for name in param_names):
        return [".to_k.", ".to_v."]
    return []


def generator_build(
    generator: torch.nn.Module,
    mode: str,
    *,
    lora_rank: int = 0,
    lora_target_modules: Sequence[str] | None = None,
    kv_patterns: Sequence[str] | None = None,
    lora_alpha: int | None = None,
    init_lora_weights: str = "gaussian",
) -> tuple[torch.nn.Module, list[torch.nn.Parameter]]:
    mode = mode.lower().strip()
    if mode not in {"none", "lora", "kv", "full"}:
        raise ValueError(
            f"Unsupported generator mode: {mode}. Expected one of "
            "['none', 'lora', 'kv', 'full']."
        )

    generator.requires_grad_(False)

    if mode == "none":
        return generator, []

    if mode == "full":
        generator.train().requires_grad_(True)
        return generator, [p for p in generator.parameters() if p.requires_grad]

    if mode == "kv":
        patterns = (
            list(kv_patterns)
            if kv_patterns is not None
            else _default_kv_patterns(generator)
        )
        if not patterns:
            raise ValueError(
                "Could not infer K/V parameter patterns. Pass kv_patterns explicitly."
            )
        for name, param in generator.named_parameters():
            if any(pattern in name for pattern in patterns):
                param.requires_grad_(True)
        params_to_optimize = [p for p in generator.parameters() if p.requires_grad]
        if not params_to_optimize:
            raise ValueError("No trainable K/V parameters found for generator.")
        return generator, params_to_optimize

    if lora_rank <= 0:
        raise ValueError("--generator_finetune=lora requires lora_rank > 0.")
    if not hasattr(generator, "add_adapter"):
        raise ValueError(
            "Generator does not expose add_adapter; LoRA mode is unsupported for this module."
        )

    target_modules = (
        list(lora_target_modules)
        if lora_target_modules is not None
        else _default_lora_targets(generator)
    )
    lora_config = LoraConfig(
        r=int(lora_rank),
        lora_alpha=int(lora_alpha if lora_alpha is not None else lora_rank),
        init_lora_weights=init_lora_weights,
        target_modules=target_modules,
    )
    generator.add_adapter(lora_config)
    params_to_optimize = [p for p in generator.parameters() if p.requires_grad]
    if not params_to_optimize:
        raise ValueError("No trainable LoRA parameters found after add_adapter.")
    return generator, params_to_optimize
