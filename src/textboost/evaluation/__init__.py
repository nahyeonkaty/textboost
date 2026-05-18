"""
Evaluation module for text-to-image personalization methods.

This module provides comprehensive evaluation functionality including:
- Core evaluation planning and organization
- Unified evaluation runner
- Metrics computation (DINO, VQA, etc.)
- Instance and prompt management
"""

# Core functionality
from .core import (
    INSTANCES,
    DEFAULT_PROMPTS,
    load_prompts_from_file,
    get_prompts,
    discover_instances,
    create_safe_filename,
    EvaluationPlan,
    EvaluationPlanner,
    print_plan_summary,
    print_batch_summary,
)

# Unified runner
from .runner import UnifiedEvaluationRunner, quick_evaluate, HAS_METRICS

# Try to import metrics functionality
try:
    from .metrics import EvaluationSuite, EvaluationConfig

    __all__ = [
        # Core
        "INSTANCES",
        "DEFAULT_PROMPTS",
        "load_prompts_from_file",
        "get_prompts",
        "discover_instances",
        "create_safe_filename",
        "EvaluationPlan",
        "EvaluationPlanner",
        "print_plan_summary",
        "print_batch_summary",
        # Runner
        "UnifiedEvaluationRunner",
        "quick_evaluate",
        "HAS_METRICS",
        # Metrics
        "EvaluationSuite",
        "EvaluationConfig",
    ]
except ImportError:
    __all__ = [
        # Core
        "INSTANCES",
        "DEFAULT_PROMPTS",
        "load_prompts_from_file",
        "get_prompts",
        "discover_instances",
        "create_safe_filename",
        "EvaluationPlan",
        "EvaluationPlanner",
        "print_plan_summary",
        "print_batch_summary",
        # Runner
        "UnifiedEvaluationRunner",
        "quick_evaluate",
        "HAS_METRICS",
    ]

# Try to import dreambooth functionality for backward compatibility
try:
    from .dreambooth import *
except ImportError:
    pass
