from __future__ import annotations

import torch
import torch.nn as nn

from textboost.adapters import Adapter


def make_expand_adapter_key(layer_name: str) -> str:
    return layer_name.replace(".", "__")


def iter_cross_attention_to_k_layers(model: nn.Module):
    index = 0
    for name, module in model.named_modules():
        if "attn2.to_k" in name:
            yield index, name, module
            index += 1


class ExpandAdapterBank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict()

    def add_adapter(
        self,
        key: str,
        *,
        dim: int,
        rank: int,
        bias: bool = False,
    ) -> None:
        if key in self.adapters:
            return
        self.adapters[key] = Adapter(dim, rank, bias=bias)

    def add_layer(
        self,
        layer_name: str,
        *,
        dim: int,
        rank: int,
        bias: bool = False,
    ) -> str:
        key = make_expand_adapter_key(layer_name)
        self.add_adapter(key, dim=dim, rank=rank, bias=bias)
        return key

    def compute_expanded_hidden_states(
        self,
        hidden_states: torch.Tensor,
        adapter_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if adapter_mask is None:
            return {}
        return {
            key: adapter(hidden_states, adapter_mask)
            for key, adapter in self.adapters.items()
        }


def attach_expand_bank_to_unet(
    unet: nn.Module,
    expand_bank: ExpandAdapterBank,
    *,
    rank: int | None = None,
    bias: bool = False,
    layer_indices: set[int] | None = None,
    create_missing: bool = True,
) -> list[tuple[int, str, str]]:
    attached_layers: list[tuple[int, str, str]] = []
    for index, name, module in iter_cross_attention_to_k_layers(unet):
        if layer_indices is not None and index not in layer_indices:
            continue

        key = make_expand_adapter_key(name)
        if create_missing:
            if rank is None:
                raise ValueError("`rank` must be provided when create_missing=True.")
            if not hasattr(module, "in_features"):
                raise ValueError(
                    f"Layer {name} does not expose in_features for expand adapter setup."
                )
            expand_bank.add_adapter(
                key,
                dim=int(module.in_features),
                rank=rank,
                bias=bias,
            )

        if key in expand_bank.adapters:
            setattr(module, "_textboost_expand_key", key)
            attached_layers.append((index, name, key))

    return attached_layers


def build_expand_bank_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> ExpandAdapterBank:
    adapter_keys = sorted(
        {
            key.split(".")[1]
            for key in state_dict
            if key.startswith("adapters.") and key.endswith(".down.weight")
        }
    )
    if not adapter_keys:
        raise ValueError("No expand bank adapter weights found in state_dict.")

    expand_bank = ExpandAdapterBank()
    for adapter_key in adapter_keys:
        down_weight_key = f"adapters.{adapter_key}.down.weight"
        up_bias_key = f"adapters.{adapter_key}.up.bias"
        down_weight = state_dict[down_weight_key]
        rank, dim = down_weight.shape
        expand_bank.add_adapter(
            adapter_key,
            dim=int(dim),
            rank=int(rank),
            bias=up_bias_key in state_dict,
        )

    incompatible = expand_bank.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Invalid expand bank state_dict. "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return expand_bank


def export_expand_bank_state_dict(
    expand_bank: ExpandAdapterBank,
) -> dict[str, torch.Tensor]:
    state_dict = expand_bank.state_dict()
    return {
        key: value.detach().cpu().clone().float() for key, value in state_dict.items()
    }
