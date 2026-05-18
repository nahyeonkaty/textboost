"""Custom CLIP text encoder wrappers with adapter-mask-aware LoRA support.

These wrappers extend HuggingFace's CLIPTextModel and CLIPTextModelWithProjection
to support masked LoRA adaptation, where LoRA outputs are scaled by an adapter_mask
(typically active only on newly added placeholder tokens).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from peft.tuners.lora import Linear as LoraLinear
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from textboost.expand_bank import ExpandAdapterBank


def build_adapter_mask_from_inputs(
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    placeholder_token_id_threshold: int = 49407,
    pad_token_id: int | None = None,
    eos_token_id: int | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor | None:
    """Build adapter mask from token ids and attention mask.

    Mask rule:
    - activate from first placeholder/new token onward (id > threshold),
    - keep padding masked out,
    - optionally mask EOS token(s) out.
    """
    if input_ids is None:
        return None

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    if attention_mask is None:
        if pad_token_id is not None:
            attention_mask = (input_ids != pad_token_id).to(dtype=torch.float32)
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
    elif attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)

    unknown_tokens = input_ids > placeholder_token_id_threshold
    has_unknown = unknown_tokens.any(dim=1, keepdim=True)
    first_unknown = torch.argmax(unknown_tokens.long(), dim=1, keepdim=True)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

    adapter_mask = positions >= first_unknown
    adapter_mask = adapter_mask & has_unknown & attention_mask.bool()

    if eos_token_id is not None:
        if isinstance(eos_token_id, (list, tuple)):
            non_eos_mask = torch.ones_like(input_ids, dtype=torch.bool)
            for eos_id in eos_token_id:
                non_eos_mask = non_eos_mask & (input_ids != int(eos_id))
        else:
            non_eos_mask = input_ids != int(eos_token_id)
        adapter_mask = adapter_mask & non_eos_mask

    return adapter_mask


class TextBoostCLIPMixin:
    """Mixin that adds adapter-mask-aware LoRA methods to a CLIP text encoder."""

    _adapter_mask: torch.Tensor | None = None
    _placeholder_token_id_threshold: int = 49407
    _last_expand_hidden_states: dict[str, torch.Tensor] | None = None
    _expand_adapter_bank: ExpandAdapterBank | None

    def set_adapter_mask(self) -> None:
        """Register adapter mask storage on all LoRA layers."""
        for _name, module in self.named_modules():
            if isinstance(module, LoraLinear):
                module._adapter_mask = None

    def replace_lora_forward(self, verbose: bool = False) -> None:
        """Replace each LoRA layer's forward so that its LoRA output is
        element-wise multiplied by the current adapter_mask before being
        added to the base-layer output.
        """
        for _name, module in self.named_modules():
            if isinstance(module, LoraLinear):
                _patch_lora_forward(module, verbose=verbose)

    def _set_adapter_mask_on_lora_layers(
        self, adapter_mask: torch.Tensor | None
    ) -> None:
        """Propagate adapter_mask to all LoRA layers."""
        for module in self.modules():
            if isinstance(module, LoraLinear):
                module._adapter_mask = adapter_mask

    def _extract_input_and_attention_from_forward(
        self, args, kwargs
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")

        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if attention_mask is None and len(args) > 1:
            attention_mask = args[1]
        return input_ids, attention_mask

    def _build_adapter_mask_from_inputs(
        self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        config = getattr(self, "config", None)
        return build_adapter_mask_from_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            placeholder_token_id_threshold=self._placeholder_token_id_threshold,
            pad_token_id=getattr(config, "pad_token_id", None),
        )

    def _resolve_adapter_mask(
        self, args, kwargs, adapter_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        if adapter_mask is not None:
            return adapter_mask
        input_ids, attention_mask = self._extract_input_and_attention_from_forward(
            args, kwargs
        )
        return self._build_adapter_mask_from_inputs(input_ids, attention_mask)

    def set_expand_adapter_bank(
        self,
        expand_adapter_bank: ExpandAdapterBank | None,
    ) -> None:
        self._expand_adapter_bank = expand_adapter_bank

    def get_expand_adapter_bank(self) -> ExpandAdapterBank | None:
        expand_adapter_bank = getattr(self, "_expand_adapter_bank", None)
        if isinstance(expand_adapter_bank, ExpandAdapterBank):
            return expand_adapter_bank
        return None

    def create_adapters(
        self,
        *,
        num_layers: int,
        rank: int | None = None,
        input_dim: int | None = None,
        input_dims: Sequence[int] | None = None,
        keys: Sequence[str] | None = None,
        bias: bool = False,
    ) -> ExpandAdapterBank:
        if num_layers < 0:
            raise ValueError("`num_layers` must be >= 0.")
        if rank is None:
            rank = 1
        if rank <= 0:
            raise ValueError("`rank` must be > 0.")
        if keys is not None and len(keys) != num_layers:
            raise ValueError("`keys` length must match `num_layers`.")
        if input_dims is not None and len(input_dims) != num_layers:
            raise ValueError("`input_dims` length must match `num_layers`.")

        layer_dims: list[int] = []
        if num_layers > 0:
            if input_dims is not None:
                layer_dims = [int(dim) for dim in input_dims]
            else:
                if input_dim is None:
                    config = getattr(self, "config", None)
                    input_dim = getattr(config, "hidden_size", None)
                if input_dim is None:
                    raise ValueError(
                        "`input_dim` or `input_dims` must be provided when `num_layers` > 0."
                    )
                layer_dims = [int(input_dim)] * num_layers

        expand_adapter_bank = self.get_expand_adapter_bank()
        if expand_adapter_bank is None:
            expand_adapter_bank = ExpandAdapterBank()
            self.set_expand_adapter_bank(expand_adapter_bank)

        if num_layers == 0:
            return expand_adapter_bank

        layer_keys = (
            list(keys)
            if keys is not None
            else [f"layer_{i}" for i in range(num_layers)]
        )
        for key, dim in zip(layer_keys, layer_dims):
            expand_adapter_bank.add_adapter(
                key,
                dim=dim,
                rank=int(rank),
                bias=bias,
            )
        return expand_adapter_bank

    def create_expand_adapters(
        self,
        *,
        num_layers: int,
        rank: int | None = None,
        input_dim: int | None = None,
        input_dims: Sequence[int] | None = None,
        keys: Sequence[str] | None = None,
        bias: bool = False,
    ) -> ExpandAdapterBank:
        return self.create_adapters(
            num_layers=num_layers,
            rank=rank,
            input_dim=input_dim,
            input_dims=input_dims,
            keys=keys,
            bias=bias,
        )

    def pop_last_expand_hidden_states(self) -> dict[str, torch.Tensor] | None:
        expanded = self._last_expand_hidden_states
        self._last_expand_hidden_states = None
        return expanded

    class _TextBoostEncoderOutput:
        def __init__(
            self,
            base_output: Any,
            expand_hidden_state: dict[str, torch.Tensor] | None,
        ) -> None:
            self._base_output = base_output
            self.expand_hidden_state = expand_hidden_state

        @property
        def last_hidden_state(self) -> torch.Tensor:
            if hasattr(self._base_output, "last_hidden_state"):
                return self._base_output.last_hidden_state
            return self._base_output[0]

        def __getitem__(self, key):
            if key == "expand_hidden_state":
                return self.expand_hidden_state
            return self._base_output[key]

        def __iter__(self):
            return iter(self._base_output)

        def __len__(self) -> int:
            return len(self._base_output)

        def __getattr__(self, name: str):
            return getattr(self._base_output, name)

    def _extract_last_hidden_from_outputs(self, outputs: Any) -> torch.Tensor:
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            return outputs.last_hidden_state
        return outputs[0]

    def _extract_penultimate_hidden_from_outputs(
        self,
        outputs: Any,
    ) -> torch.Tensor | None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None and isinstance(outputs, tuple) and len(outputs) >= 3:
            hidden_states = outputs[2]
        if hidden_states is None or len(hidden_states) < 2:
            return None
        return hidden_states[-2]

    def _compute_expand_hidden_states(
        self,
        outputs: Any,
        *,
        expand_adapter_bank: ExpandAdapterBank | None,
        expand_adapter_mask: torch.Tensor | None,
        expand_prefix_hidden_states: torch.Tensor | None,
        expand_hidden_states_override: torch.Tensor | None,
    ) -> dict[str, torch.Tensor] | None:
        self._last_expand_hidden_states = None
        if expand_adapter_bank is None:
            expand_adapter_bank = self.get_expand_adapter_bank()
        if expand_adapter_bank is None:
            return None

        if expand_hidden_states_override is not None:
            hidden_for_expand = expand_hidden_states_override
        else:
            hidden_for_expand = self._extract_penultimate_hidden_from_outputs(outputs)
            if hidden_for_expand is None:
                hidden_for_expand = self._extract_last_hidden_from_outputs(outputs)
            if expand_prefix_hidden_states is not None:
                hidden_for_expand = torch.cat(
                    [expand_prefix_hidden_states, hidden_for_expand], dim=-1
                )

        expand_hidden_states = expand_adapter_bank.compute_expanded_hidden_states(
            hidden_for_expand,
            expand_adapter_mask,
        )
        if not expand_hidden_states:
            return None
        self._last_expand_hidden_states = expand_hidden_states
        return expand_hidden_states


def _patch_lora_forward(lora_module: LoraLinear, verbose: bool = False) -> None:
    """Monkey-patch a PEFT LoraLinear module so that its LoRA delta
    is element-wise masked by ``lora_module._adapter_mask``.
    """
    if getattr(lora_module, "_textboost_mask_patch_applied", False):
        return
    original_forward = lora_module.forward

    def masked_forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Base layer result (without LoRA).
        base_result = lora_module.base_layer(x)

        # Compute LoRA delta.
        lora_delta = original_forward(x, *args, **kwargs) - base_result

        mask = getattr(lora_module, "_adapter_mask", None)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            lora_delta = (
                mask.to(device=lora_delta.device, dtype=lora_delta.dtype) * lora_delta
            )

        return base_result + lora_delta

    lora_module.forward = masked_forward
    lora_module._textboost_mask_patch_applied = True
    if verbose:
        print(f"  Patched LoRA forward: {lora_module}")


class TextModel(TextBoostCLIPMixin, CLIPTextModel):
    """CLIPTextModel with adapter-mask-aware LoRA support.

    The forward method accepts an additional ``adapter_mask`` keyword argument
    that is propagated to all LoRA layers before running the standard forward.
    """

    def forward(
        self,
        *args,
        adapter_mask: torch.Tensor | None = None,
        expand_adapter_bank: ExpandAdapterBank | None = None,
        expand_adapter_mask: torch.Tensor | None = None,
        expand_prefix_hidden_states: torch.Tensor | None = None,
        expand_hidden_states_override: torch.Tensor | None = None,
        **kwargs,
    ):
        resolved_mask = self._resolve_adapter_mask(args, kwargs, adapter_mask)
        self._set_adapter_mask_on_lora_layers(resolved_mask)
        outputs = super().forward(*args, **kwargs)
        expand_hidden_states = self._compute_expand_hidden_states(
            outputs,
            expand_adapter_bank=expand_adapter_bank,
            expand_adapter_mask=(
                expand_adapter_mask
                if expand_adapter_mask is not None
                else resolved_mask
            ),
            expand_prefix_hidden_states=expand_prefix_hidden_states,
            expand_hidden_states_override=expand_hidden_states_override,
        )
        return self._TextBoostEncoderOutput(outputs, expand_hidden_states)


class TextModelWithProjection(TextBoostCLIPMixin, CLIPTextModelWithProjection):
    """CLIPTextModelWithProjection with adapter-mask-aware LoRA support.

    Same API as :class:`TextModel` but wraps the projection variant used
    as the second text encoder in SDXL.
    """

    def forward(
        self,
        *args,
        adapter_mask: torch.Tensor | None = None,
        expand_adapter_bank: ExpandAdapterBank | None = None,
        expand_adapter_mask: torch.Tensor | None = None,
        expand_prefix_hidden_states: torch.Tensor | None = None,
        expand_hidden_states_override: torch.Tensor | None = None,
        **kwargs,
    ):
        resolved_mask = self._resolve_adapter_mask(args, kwargs, adapter_mask)
        self._set_adapter_mask_on_lora_layers(resolved_mask)
        outputs = super().forward(*args, **kwargs)
        expand_hidden_states = self._compute_expand_hidden_states(
            outputs,
            expand_adapter_bank=expand_adapter_bank,
            expand_adapter_mask=(
                expand_adapter_mask
                if expand_adapter_mask is not None
                else resolved_mask
            ),
            expand_prefix_hidden_states=expand_prefix_hidden_states,
            expand_hidden_states_override=expand_hidden_states_override,
        )
        return self._TextBoostEncoderOutput(outputs, expand_hidden_states)
