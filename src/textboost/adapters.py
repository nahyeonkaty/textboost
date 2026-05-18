from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TrfConfig:
    name: str = "default"
    r: int = 1
    target_index: int | list[int] | None = None
    target_modules: list[str] | None = None
    exclude_modules: list[str] | None = None


class Adapter(nn.Module):
    def __init__(self, dim: int, rank: int, bias: bool = False) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=bias)
        nn.init.normal_(self.down.weight, std=(1 / rank))
        nn.init.zeros_(self.up.weight)
        if bias:
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        identity = x
        x = x.to(dtype=self.down.weight.dtype)
        x = self.up(self.down(x))
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        x = mask.to(dtype=x.dtype) * x
        x = x.to(dtype)
        return identity + x


class TrfLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        config: TrfConfig,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = config.r
        self.target_index = config.target_index
        self.adapter_mask: torch.Tensor | None = None

        self.enabled_adapters: list[str] = []
        self.down = nn.ModuleDict()
        self.up = nn.ModuleDict()
        self.add_adapter(config.name)

    def add_adapter(self, name: str) -> None:
        if name in self.down or name in self.up:
            raise ValueError(f"Adapter with name '{name}' already exists.")
        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features
        self.down[name] = nn.Linear(in_features, self.rank, bias=False)
        self.up[name] = nn.Linear(self.rank, out_features, bias=False)
        nn.init.kaiming_normal_(self.down[name].weight)
        nn.init.zeros_(self.up[name].weight)
        self.enabled_adapters.append(name)

    def set_grad_enabled(self, enabled: bool = True) -> None:
        self.down.requires_grad_(enabled)
        self.up.requires_grad_(enabled)

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        state_dict: dict[str, torch.Tensor] = {}
        for name, module in self.down.items():
            state_dict[f"down.{name}.weight"] = module.weight
        for name, module in self.up.items():
            state_dict[f"up.{name}.weight"] = module.weight
            if module.bias is not None:
                state_dict[f"up.{name}.bias"] = module.bias
        return state_dict

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        base_output = self.base_layer(x, *args, **kwargs)

        adapter_output = 0
        for k in self.enabled_adapters:
            lora_output = self.up[k](self.down[k](x))
            lora_output = self.adapter_mask * lora_output
            adapter_output = adapter_output + lora_output
        output = base_output + adapter_output
        return output


def print_trainable_parameters(model: nn.Module) -> None:
    """Print information about trainable parameters."""
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")


def attach_adapters_to_model(
    model: nn.Module,
    config: TrfConfig,
) -> nn.Module:
    model.eval().requires_grad_(False)

    # Replace target modules with TrfLayer.
    target_modules = config.target_modules
    exclude_modules = config.exclude_modules or []
    modules_to_replace: list[tuple[str, tuple[nn.Module, list[str]]]] = []
    for name, module in model.named_modules():
        if any(name.endswith(target) for target in target_modules) and not any(
            name.endswith(exclude) for exclude in exclude_modules
        ):
            # Navigate to the parent module
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            modules_to_replace.append((name, (parent, parts)))

    for name, (parent, parts) in modules_to_replace:
        print(f"Replacing module: {name}")
        original_module = getattr(parent, parts[-1])
        custom_module = TrfLayer(original_module, config)
        setattr(parent, parts[-1], custom_module)

    # Register pre-forward hook to set adapter_mask.
    def pre_hook(
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        causal: bool = True,
    ) -> None:
        indices = input[0]
        target = config.target_index
        if isinstance(target, int):
            mask = (indices == target).float()
        elif isinstance(target, (list, tuple)):
            mask = torch.zeros_like(indices, dtype=torch.float)
            for t in target:
                mask = mask + (indices == t).float()
            mask = mask.clamp(max=1.0)
        else:
            raise ValueError("config.target_index must be int or list/tuple of ints.")
        if causal:
            causal_mask = torch.zeros_like(indices, dtype=torch.float)
            first_positions = mask.argmax(dim=1)  # (batch_size,)
            for i, pos in enumerate(first_positions):
                causal_mask[i, pos:] = 1.0
            mask = causal_mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        for _name, submodule in module.named_modules():
            if isinstance(submodule, TrfLayer):
                setattr(submodule, "adapter_mask", mask)

    model.register_forward_pre_hook(pre_hook)

    # Register state_dict hook to save only adapter parameters.
    def state_dict_hook(
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
    ) -> None:
        modules_to_keep: list[str] = []
        for name, mod in module.named_modules():
            if isinstance(mod, TrfLayer):
                adapter_state_dict = mod.adapter_state_dict()
                modules_to_keep.extend(
                    [f"{name}.{k}" for k in adapter_state_dict.keys()]
                )
        for key in list(state_dict.keys()):
            if not any(key.endswith(keep) for keep in modules_to_keep):
                del state_dict[key]

    model.register_state_dict_post_hook(state_dict_hook)

    return model
