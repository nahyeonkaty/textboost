#!/usr/bin/env python3
"""
Prototype test for a clean CPA implementation.

This script does NOT modify current production code paths. It validates that a
subclass-based CPA LoRA layer (no monkey patching) has the expected behavior:
1) no mask -> identical to standard LoRA
2) zero mask -> base layer only
3) one mask -> full LoRA effect
4) partial mask -> base + mask * delta
5) parity with current monkey-patched forward in textboost.text_encoders.clip
"""

from __future__ import annotations

import copy
import math

import torch
from peft.tuners.lora.layer import Linear as LoraLinear

from textboost.text_encoders.clip import _patch_lora_forward


class CPALoraLinear(LoraLinear):
    """Clean prototype: masking implemented directly in forward."""

    def forward(
        self,
        x: torch.Tensor,
        *args,
        adapter_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        full = super().forward(x, *args, **kwargs)
        if adapter_mask is None:
            return full

        base = self.base_layer(x)
        delta = full - base

        mask = adapter_mask
        if mask.ndim == 2 and delta.ndim >= 3:
            mask = mask.unsqueeze(-1)
        mask = mask.to(device=delta.device, dtype=delta.dtype)
        return base + mask * delta


def _make_triplet(
    in_dim: int = 32, out_dim: int = 32, r: int = 4, seed: int = 0
) -> tuple[LoraLinear, LoraLinear, CPALoraLinear]:
    torch.manual_seed(seed)
    base = torch.nn.Linear(in_dim, out_dim, bias=False)

    ref = LoraLinear(
        copy.deepcopy(base),
        adapter_name="default",
        r=r,
        lora_alpha=r,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    patched = LoraLinear(
        copy.deepcopy(base),
        adapter_name="default",
        r=r,
        lora_alpha=r,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    cpa = CPALoraLinear(
        copy.deepcopy(base),
        adapter_name="default",
        r=r,
        lora_alpha=r,
        lora_dropout=0.0,
        init_lora_weights=True,
    )

    patched.load_state_dict(ref.state_dict(), strict=False)
    cpa.load_state_dict(ref.state_dict(), strict=False)
    _patch_lora_forward(patched, verbose=False)

    ref.eval()
    patched.eval()
    cpa.eval()
    return ref, patched, cpa


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6):
    max_err = (a - b).abs().max().item()
    if not math.isclose(max_err, 0.0, abs_tol=atol):
        raise AssertionError(f"{name} failed: max_err={max_err:.3e}, atol={atol}")
    print(f"[PASS] {name} (max_err={max_err:.3e})")


def test_sequence_shape():
    ref, patched, cpa = _make_triplet(seed=7)

    x = torch.randn(2, 5, 32)
    mask = (torch.rand(2, 5) > 0.4).float()

    full_ref = ref(x)
    base = cpa.base_layer(x)
    out_no_mask = cpa(x)
    out_zero = cpa(x, adapter_mask=torch.zeros_like(mask))
    out_one = cpa(x, adapter_mask=torch.ones_like(mask))
    out_mask = cpa(x, adapter_mask=mask)
    expected = base + mask.unsqueeze(-1) * (full_ref - base)

    _assert_close("3D/no-mask==ref", out_no_mask, full_ref)
    _assert_close("3D/zero-mask==base", out_zero, base)
    _assert_close("3D/one-mask==ref", out_one, full_ref)
    _assert_close("3D/partial-mask equation", out_mask, expected)

    # Parity with current monkey-patched behavior.
    patched._adapter_mask = mask
    out_patch = patched(x)
    _assert_close("3D/parity with monkey-patch", out_mask, out_patch)


def test_vector_shape():
    ref, patched, cpa = _make_triplet(seed=11)

    x = torch.randn(4, 32)
    mask = (torch.rand(4, 1) > 0.5).float()

    full_ref = ref(x)
    base = cpa.base_layer(x)
    out_mask = cpa(x, adapter_mask=mask)
    expected = base + mask * (full_ref - base)

    _assert_close("2D/partial-mask equation", out_mask, expected)

    patched._adapter_mask = mask
    out_patch = patched(x)
    _assert_close("2D/parity with monkey-patch", out_mask, out_patch)


def main():
    test_sequence_shape()
    test_vector_shape()
    print("All clean CPA prototype tests passed.")


if __name__ == "__main__":
    main()
