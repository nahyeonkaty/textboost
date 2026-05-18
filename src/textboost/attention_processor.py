from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, SanaLinearAttnProcessor2_0

from textboost.expand_bank import make_expand_adapter_key


def custom_lora_forward(
    self,
    x: torch.Tensor,
    adapter_mask: torch.Tensor | None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        raise NotImplementedError
    elif self.merged:
        raise NotImplementedError
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                lora_result = lora_B(lora_A(dropout(x))) * scaling
            else:
                x = dropout(x)
                lora_result = self.lora_magnitude_vector[active_adapter](
                    x,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    base_layer=self.get_base_layer(),
                )
            if adapter_mask is not None:
                lora_result = lora_result * adapter_mask
            result = result + lora_result

        result = result.to(torch_result_dtype)

    return result


class TextBoostAttnProcessor(nn.Module):
    def __init__(
        self,
        expand_key: str | None = None,
        expand_fallback_key: str | None = None,
    ) -> None:
        super().__init__()
        self.expand_key = expand_key
        self.expand_fallback_key = expand_fallback_key

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        adapter_mask: torch.Tensor | None = None,
        expand_hidden_states: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

        if hasattr(attn.to_k, "base_layer"):  # if LoRA
            if adapter_mask is not None:
                if adapter_mask.ndim == 2:
                    adapter_mask = adapter_mask.unsqueeze(-1)
            key = custom_lora_forward(
                attn.to_k, encoder_hidden_states, adapter_mask=adapter_mask
            )
            value = custom_lora_forward(
                attn.to_v, encoder_hidden_states, adapter_mask=adapter_mask
            )
        elif hasattr(attn.to_k, "adapter"):
            layer_input = attn.to_k.adapter(encoder_hidden_states, adapter_mask)
            key = attn.to_k(layer_input)
            value = attn.to_v(layer_input)
        elif expand_hidden_states is not None:
            layer_key = self.expand_key
            if layer_key is None and hasattr(attn.to_k, "_textboost_expand_key"):
                layer_key = getattr(attn.to_k, "_textboost_expand_key")
            layer_input = encoder_hidden_states
            if layer_key is not None and layer_key in expand_hidden_states:
                layer_input = expand_hidden_states[layer_key]
            elif (
                self.expand_fallback_key is not None
                and self.expand_fallback_key in expand_hidden_states
            ):
                layer_input = expand_hidden_states[self.expand_fallback_key]
            if layer_key is not None or self.expand_fallback_key is not None:
                key = attn.to_k(layer_input)
                value = attn.to_v(layer_input)
            else:
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if crossattn:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
        else:
            attention_probs = attn.get_attention_scores(query, key, None)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SanaAttnProcessor(nn.Module):
    def __init__(
        self,
        expand_key: str | None = None,
        expand_fallback_key: str | None = None,
    ) -> None:
        super().__init__()
        self.linear_attn_processor = SanaLinearAttnProcessor2_0()
        self.adapter_mask: torch.Tensor | None = None
        self.expand_key = expand_key
        self.expand_fallback_key = expand_fallback_key

    def set_adapter_mask(self, adapter_mask: torch.Tensor) -> None:
        self.adapter_mask = adapter_mask

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        adapter_mask: torch.Tensor | None = None,
        expand_hidden_states: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not attn.is_cross_attention:
            return self.linear_attn_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if adapter_mask is None:
            adapter_mask = self.adapter_mask
        if adapter_mask is not None and adapter_mask.ndim == 2:
            adapter_mask = adapter_mask.unsqueeze(-1)

        if hasattr(attn.to_k, "base_layer"):  # if LoRA
            key = custom_lora_forward(
                attn.to_k, encoder_hidden_states, adapter_mask=adapter_mask
            )
            value = custom_lora_forward(
                attn.to_v, encoder_hidden_states, adapter_mask=adapter_mask
            )
        elif hasattr(attn.to_k, "adapter"):
            layer_input = attn.to_k.adapter(encoder_hidden_states, adapter_mask)
            key = attn.to_k(layer_input)
            value = attn.to_v(layer_input)
        elif expand_hidden_states is not None:
            layer_key = self.expand_key
            if layer_key is None and hasattr(attn.to_k, "_textboost_expand_key"):
                layer_key = getattr(attn.to_k, "_textboost_expand_key")
            layer_input = encoder_hidden_states
            if layer_key is not None and layer_key in expand_hidden_states:
                layer_input = expand_hidden_states[layer_key]
            elif (
                self.expand_fallback_key is not None
                and self.expand_fallback_key in expand_hidden_states
            ):
                layer_input = expand_hidden_states[self.expand_fallback_key]
            if layer_key is not None or self.expand_fallback_key is not None:
                key = attn.to_k(layer_input)
                value = attn.to_v(layer_input)
            else:
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def build_textboost_attn_processors(
    unet: nn.Module,
    *,
    use_sana: bool = False,
) -> dict[str, nn.Module]:
    processor_cls = SanaAttnProcessor if use_sana else TextBoostAttnProcessor
    processors: dict[str, nn.Module] = {}
    cross_attn_index = 0
    for name in unet.attn_processors.keys():
        expand_key = None
        expand_fallback_key = None
        if ".attn2.processor" in name:
            layer_name = name.replace(".processor", ".to_k")
            expand_key = make_expand_adapter_key(layer_name)
            expand_fallback_key = f"layer_{cross_attn_index}"
            cross_attn_index += 1
        processors[name] = processor_cls(
            expand_key=expand_key,
            expand_fallback_key=expand_fallback_key,
        )
    return processors
