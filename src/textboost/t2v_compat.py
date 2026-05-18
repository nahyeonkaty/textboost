"""Compatibility helpers for importing t2v_metrics on newer transformers."""

from __future__ import annotations

import importlib
import sys
import types
from typing import List, Tuple


_MOVED_MODELING_UTILS = (
    "apply_chunking_to_forward",
    "find_pruneable_heads_and_indices",
    "prune_linear_layer",
)


def _patch_transformers_modeling_utils() -> None:
    """Backfill helpers moved from modeling_utils to pytorch_utils in transformers."""
    try:
        from transformers import modeling_utils, pytorch_utils
    except Exception:
        return

    for name in _MOVED_MODELING_UTILS:
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))


def _install_clip_stub_if_missing() -> bool:
    """Install a minimal clip stub if package is missing.

    t2v_metrics imports ITM/ImageReward symbols at package import time, which can
    require `clip` even when only VQAScore is used. The stub keeps VQA usable.
    """
    try:
        importlib.import_module("clip")
        return False
    except ModuleNotFoundError:
        pass

    def _missing_clip(*args, **kwargs):
        raise ImportError(
            "Optional dependency `clip` is required by ImageReward-based ITM "
            "components in t2v_metrics. Install it to use those models."
        )

    clip_stub = types.ModuleType("clip")
    clip_stub.__textboost_stub__ = True
    clip_stub.load = _missing_clip
    clip_stub.tokenize = _missing_clip
    clip_stub.model = types.SimpleNamespace(convert_weights=_missing_clip)
    sys.modules["clip"] = clip_stub
    return True


def import_t2v_metrics() -> Tuple[object, List[str]]:
    """Import t2v_metrics with compatibility patches applied."""
    notes: List[str] = []
    _patch_transformers_modeling_utils()

    if _install_clip_stub_if_missing():
        notes.append(
            "Loaded t2v_metrics without `clip`; VQAScore and t2v CLIPScore are "
            "available, but ImageReward-based ITM models are disabled."
        )

    module = importlib.import_module("t2v_metrics")
    return module, notes
