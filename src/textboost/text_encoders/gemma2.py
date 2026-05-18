"""Custom Gemma2 text encoder wrapper with adapter-mask-aware LoRA support.

This wrapper extends HuggingFace's Gemma2Model to support masked LoRA
adaptation, where LoRA outputs are scaled by an adapter_mask (typically
active only on newly added placeholder tokens).
"""

from __future__ import annotations

import torch
from transformers import Gemma2Model

from textboost.expand_bank import ExpandAdapterBank
from textboost.text_encoders.clip import TextBoostCLIPMixin


class TextModel(TextBoostCLIPMixin, Gemma2Model):
    """Gemma2Model with adapter-mask-aware LoRA support.

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
