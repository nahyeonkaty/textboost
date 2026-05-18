"""
Core evaluation functionality for text-to-image personalization methods.

This module provides the fundamental building blocks for evaluation:
- Instance and prompt management
- Directory structure creation
- Evaluation planning
- File organization utilities
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Import instances from existing dreambooth module
try:
    from .dreambooth import INSTANCES
except ImportError:
    # Fallback instances mapping if import fails
    INSTANCES = {
        "backpack": "backpack",
        "backpack_dog": "backpack",
        "bear_plushie": "stuffed animal",
        "berry_bowl": "bowl",
        "can": "can",
        "candle": "candle",
        "cat": "cat",
        "cat2": "cat",
        "clock": "clock",
        "colorful_sneaker": "sneaker",
        "dog": "dog",
        "dog2": "dog",
        "dog3": "dog",
        "dog5": "dog",
        "dog6": "dog",
        "dog7": "dog",
        "dog8": "dog",
        "duck_toy": "toy",
        "fancy_boot": "boot",
        "grey_sloth_plushie": "stuffed animal",
        "monster_toy": "toy",
        "pink_sunglasses": "glasses",
        "poop_emoji": "toy",
        "rc_car": "toy",
        "red_cartoon": "cartoon",
        "robot_toy": "toy",
        "shiny_sneaker": "sneaker",
        "teapot": "teapot",
        "vase": "vase",
        "wolf_plushie": "stuffed animal",
    }

# Default evaluation prompts
DEFAULT_PROMPTS = [
    "a {} in the jungle",
    "a {} in the snow",
    "a {} on the beach",
    "a {} on a cobblestone street",
    "a {} wearing a red hat",
    "a {} in a chef outfit",
    "a red {}",
    "a shiny {}",
    "a {} floating on water",
    "a {} in an art gallery",
]


def load_prompts_from_file(prompts_file: str) -> List[str]:
    """Load prompts from text file."""
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise ValueError(f"No prompts found in file: {prompts_file}")

    return prompts


def get_prompts(
    prompts_file: Optional[str] = None,
    prompts_list: Optional[List[str]] = None,
    use_defaults: bool = True,
) -> List[str]:
    """Get prompts from file, list, or defaults."""
    if prompts_file:
        return load_prompts_from_file(prompts_file)
    elif prompts_list:
        return prompts_list
    elif use_defaults:
        return DEFAULT_PROMPTS
    else:
        raise ValueError("No prompts specified and defaults disabled")


def discover_instances(
    method_dir: str, specified_instances: Optional[List[str]] = None
) -> List[str]:
    """Discover available instances in the method directory."""
    method_path = Path(method_dir)
    if not method_path.exists():
        raise FileNotFoundError(f"Method directory not found: {method_dir}")

    # Get all subdirectories that match known instances
    available_instances = []
    for item in method_path.iterdir():
        if item.is_dir() and item.name in INSTANCES:
            available_instances.append(item.name)

    if specified_instances:
        # Filter to only requested instances
        invalid_instances = [
            inst for inst in specified_instances if inst not in available_instances
        ]
        if invalid_instances:
            warnings.warn(f"Requested instances not found: {invalid_instances}")

        valid_instances = [
            inst for inst in specified_instances if inst in available_instances
        ]
        if not valid_instances:
            raise ValueError("No valid instances found from the specified list")

        return valid_instances

    if not available_instances:
        raise ValueError(f"No valid instances found in {method_dir}")

    return sorted(available_instances)


def create_safe_filename(prompt: str, instance: str, max_length: int = 100) -> str:
    """Create safe filename from prompt and instance."""
    # Replace placeholder with instance class name
    formatted_prompt = prompt.replace("{}", INSTANCES[instance])
    filename = formatted_prompt.replace(" ", "_")

    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Limit length
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename + ".png"


class EvaluationPlan:
    """Represents an evaluation plan for a method."""

    def __init__(
        self,
        method_dir: str,
        instances: List[str],
        prompts: List[str],
        seeds: List[int],
        checkpoint: Optional[int] = None,
        model_type: str = "sana",
    ):
        self.method_dir = method_dir
        self.method_name = os.path.basename(method_dir)
        self.instances = instances
        self.prompts = prompts
        self.seeds = seeds
        self.checkpoint = checkpoint
        self.model_type = model_type

        # Computed properties
        self.num_instances = len(instances)
        self.num_prompts = len(prompts)
        self.num_seeds = len(seeds)
        self.total_images = self.num_instances * self.num_prompts * self.num_seeds

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "method_dir": self.method_dir,
            "method_name": self.method_name,
            "instances": self.instances,
            "num_instances": self.num_instances,
            "prompts": self.prompts,
            "num_prompts": self.num_prompts,
            "seeds": self.seeds,
            "num_seeds": self.num_seeds,
            "total_images": self.total_images,
            "checkpoint": self.checkpoint,
            "model_type": self.model_type,
        }

    def create_directory_structure(
        self, output_dir: str, overwrite: bool = False
    ) -> Dict[str, List[str]]:
        """Create directory structure and return file paths to generate."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files_to_generate = {}

        for seed in self.seeds:
            for instance in self.instances:
                instance_dir = output_path / f"seed{seed}" / instance
                instance_dir.mkdir(parents=True, exist_ok=True)

                instance_files = []
                for prompt in self.prompts:
                    filename = create_safe_filename(prompt, instance)
                    filepath = instance_dir / filename

                    # Check if file exists
                    if filepath.exists() and not overwrite:
                        continue

                    instance_files.append(str(filepath))

                if instance_files:
                    files_to_generate[f"seed{seed}/{instance}"] = instance_files

        return files_to_generate

    def save_plan(self, output_dir: str, include_file_structure: bool = True) -> str:
        """Save evaluation plan to file."""
        plan_data = {"plan": self.to_dict(), "timestamp": time.time()}

        if include_file_structure:
            files_to_generate = self.create_directory_structure(
                output_dir, overwrite=False
            )
            plan_data["files_to_generate"] = files_to_generate
            plan_data["files_count"] = sum(
                len(files) for files in files_to_generate.values()
            )

        plan_file = Path(output_dir) / "evaluation_plan.json"
        with open(plan_file, "w") as f:
            json.dump(plan_data, f, indent=2)

        return str(plan_file)


class EvaluationPlanner:
    """Creates and manages evaluation plans."""

    def __init__(
        self,
        seeds: List[int] = None,
        checkpoint: Optional[int] = None,
        model_type: str = "sana",
    ):
        self.seeds = seeds or [0, 1, 2, 3, 4]
        self.checkpoint = checkpoint
        self.model_type = model_type

    def create_plan(
        self,
        method_dir: str,
        prompts_file: Optional[str] = None,
        prompts_list: Optional[List[str]] = None,
        instances: Optional[List[str]] = None,
    ) -> EvaluationPlan:
        """Create evaluation plan for a single method."""
        # Get prompts
        prompts = get_prompts(prompts_file, prompts_list)

        # Discover instances
        discovered_instances = discover_instances(method_dir, instances)

        # Create plan
        return EvaluationPlan(
            method_dir=method_dir,
            instances=discovered_instances,
            prompts=prompts,
            seeds=self.seeds,
            checkpoint=self.checkpoint,
            model_type=self.model_type,
        )

    def create_batch_plan(
        self, method_dirs: List[str], **kwargs
    ) -> List[EvaluationPlan]:
        """Create evaluation plans for multiple methods."""
        plans = []
        for method_dir in method_dirs:
            try:
                plan = self.create_plan(method_dir, **kwargs)
                plans.append(plan)
            except Exception as e:
                warnings.warn(f"Failed to create plan for {method_dir}: {e}")

        return plans


def print_plan_summary(plan: EvaluationPlan, mode: str = "single"):
    """Print summary of evaluation plan."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATION PLAN SUMMARY - {mode.upper()}")
    print(f"{'=' * 80}")
    print(f"Method: {plan.method_name}")
    print(f"Directory: {plan.method_dir}")
    print(f"Instances: {plan.num_instances} - {plan.instances}")
    print(f"Prompts: {plan.num_prompts}")
    print(f"Seeds: {plan.num_seeds} - {plan.seeds}")
    print(f"Total images: {plan.total_images}")
    if plan.checkpoint:
        print(f"Checkpoint: {plan.checkpoint}")
    print(f"Model type: {plan.model_type}")
    print(f"{'=' * 80}")


def print_batch_summary(plans: List[EvaluationPlan]):
    """Print summary of batch evaluation plans."""
    print(f"\n{'=' * 80}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total methods: {len(plans)}")

    if plans:
        total_images = sum(plan.total_images for plan in plans)
        print(f"Total images across all methods: {total_images}")

        print(f"\nPer-method breakdown:")
        for plan in plans:
            print(
                f"  {plan.method_name}: {plan.total_images} images "
                f"({plan.num_instances} instances × {plan.num_prompts} prompts × {plan.num_seeds} seeds)"
            )

    print(f"{'=' * 80}")


# Make key functions available at module level
__all__ = [
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
]
