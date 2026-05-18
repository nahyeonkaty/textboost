"""
Improved evaluation metrics for text-to-image generation models.

This module provides efficient and user-friendly evaluation metrics including:
1. DINOv2 feature similarity
2. VQA scores using t2v_metrics

Key improvements:
- Modular design for easy extension
- Efficient batch processing
- Better error handling
- Clear progress reporting
- Configurable options
"""

import glob
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.transforms import v2
from tqdm import tqdm

from textboost.t2v_compat import import_t2v_metrics

try:
    t2v_metrics, T2V_IMPORT_NOTES = import_t2v_metrics()

    T2V_IMPORT_ERROR = None
except ImportError as e:
    t2v_metrics = None
    T2V_IMPORT_NOTES = []
    T2V_IMPORT_ERROR = e


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""

    # General settings
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4

    # DINOv2 settings
    dino_model: str = "dinov2_vitl14"
    dino_image_size: int = 224
    use_masks: bool = True

    # VQA settings
    vqa_model: str = "clip-flant5-xxl"
    vqa_batch_size: int = 16

    # Output settings
    save_individual_scores: bool = False
    verbose: bool = True


class DINOv2Evaluator:
    """DINOv2 feature similarity evaluator."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.preprocess = self._create_preprocess()

    def _create_preprocess(self):
        """Create preprocessing pipeline for DINOv2."""
        return v2.Compose(
            [
                v2.Resize((512, 512)),
                v2.Resize((self.config.dino_image_size, self.config.dino_image_size)),
                v2.ToTensor(),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _load_model(self):
        """Load DINOv2 model lazily."""
        if self.model is None:
            if self.config.verbose:
                print(f"Loading DINOv2 model: {self.config.dino_model}")
            self.model = torch.hub.load(
                "facebookresearch/dinov2", self.config.dino_model
            )
            self.model.eval().requires_grad_(False).to(self.config.device)

    def _load_reference_images(
        self, reference_dir: str, instance: str, mask_dir: Optional[str] = None
    ) -> torch.Tensor:
        """Load and preprocess reference images for an instance."""
        instance_dir = Path(reference_dir) / instance
        if not instance_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {instance_dir}")

        images = []
        image_files = list(instance_dir.glob("*.jpg")) + list(
            instance_dir.glob("*.png")
        )

        if not image_files:
            raise ValueError(f"No reference images found in {instance_dir}")

        for img_file in image_files:
            image = Image.open(img_file).convert("RGB")

            # Apply mask if available and requested
            if self.config.use_masks and mask_dir:
                mask_file = Path(mask_dir) / instance / f"{img_file.stem}.png"
                if mask_file.exists():
                    mask = Image.open(mask_file).convert("RGB")
                    mask_array = np.array(mask) / 255.0
                    image_array = np.array(image) * mask_array
                    image = Image.fromarray(image_array.astype(np.uint8))

            images.append(self.preprocess(image))

        return torch.stack(images)

    def _extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from images using DINOv2."""
        features = []

        for i in range(0, len(images), self.config.batch_size):
            batch = images[i : i + self.config.batch_size].to(self.config.device)
            with torch.no_grad():
                batch_features = self.model(batch).float().cpu().numpy()
            features.append(batch_features)

        return np.concatenate(features, axis=0)

    def evaluate_instance(
        self,
        generated_dir: str,
        reference_dir: str,
        instance: str,
        mask_dir: Optional[str] = None,
    ) -> np.ndarray:
        """Evaluate DINOv2 similarity for a single instance."""
        self._load_model()

        # Load reference images
        reference_images = self._load_reference_images(
            reference_dir, instance, mask_dir
        )
        reference_features = self._extract_features(reference_images)

        # Load generated images. Support both:
        # 1) generated_dir/seedX/instance/*.png
        # 2) generated_dir/<method>/seedX/instance/*.png
        generated_images = []
        pattern_one_level = f"{generated_dir}/*/{instance}/*.png"
        pattern_two_level = f"{generated_dir}/*/*/{instance}/*.png"
        generated_files = sorted(
            set(glob.glob(pattern_one_level) + glob.glob(pattern_two_level))
        )

        if not generated_files:
            warnings.warn(f"No generated images found for instance {instance}")
            return np.array([])

        for img_file in generated_files:
            image = Image.open(img_file).convert("RGB")
            generated_images.append(self.preprocess(image))

        if not generated_images:
            return np.array([])

        generated_images = torch.stack(generated_images)
        generated_features = self._extract_features(generated_images)

        # Compute similarity scores
        similarity_matrix = cosine_similarity(generated_features, reference_features)
        similarity_matrix[similarity_matrix < 0] = 0.0  # Clip negative similarities

        # Take max similarity for each generated image
        scores = similarity_matrix.max(axis=1)

        return scores

    def evaluate(
        self,
        generated_dir: str,
        reference_dir: str,
        instances: List[str],
        mask_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Evaluate DINOv2 similarity for multiple instances."""
        if self.config.verbose:
            print("Evaluating DINOv2 feature similarity...")

        all_scores = []
        instance_scores = {}

        for instance in tqdm(
            instances, desc="DINOv2 evaluation", disable=not self.config.verbose
        ):
            try:
                scores = self.evaluate_instance(
                    generated_dir, reference_dir, instance, mask_dir
                )
                if len(scores) > 0:
                    all_scores.append(scores)
                    instance_scores[instance] = scores

                    if self.config.verbose:
                        print(
                            f"  {instance}: {scores.mean():.3f} ± {scores.std():.3f} ({len(scores)} samples)"
                        )

            except Exception as e:
                warnings.warn(f"Failed to evaluate instance {instance}: {e}")
                instance_scores[instance] = np.array([])

        all_scores = np.concatenate(all_scores) if all_scores else np.array([])

        if self.config.verbose and len(all_scores) > 0:
            print(
                f"Overall DINOv2: {all_scores.mean():.3f} ± {all_scores.std():.3f} ({len(all_scores)} samples)"
            )

        return all_scores, instance_scores

    def cleanup(self):
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()


class VQAEvaluator:
    """VQA score evaluator using t2v_metrics."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None

    def _load_model(self):
        """Load VQA model lazily."""
        if self.model is None:
            if t2v_metrics is None:
                raise ImportError(
                    "Failed to import 't2v-metrics' required for VQA evaluation. "
                    f"Original error: {T2V_IMPORT_ERROR}. "
                    f"Compatibility notes: {T2V_IMPORT_NOTES}"
                )
            if self.config.verbose:
                print(f"Loading VQA model: {self.config.vqa_model}")
            self.model = t2v_metrics.VQAScore(
                model=self.config.vqa_model, device=self.config.device
            )
            self.model.eval().requires_grad_(False)

    def _path_to_prompt(self, path: str) -> str:
        """Convert image path to prompt text."""
        basename = Path(path).name
        return basename.replace(".png", "").replace("_", " ")

    def evaluate(self, generated_dir: str) -> Tuple[np.ndarray, List[Dict[str, str]]]:
        """Evaluate VQA scores for generated images."""
        if self.config.verbose:
            print("Evaluating VQA scores...")

        self._load_model()

        # Collect all generated images and their prompts
        image_files = glob.glob(f"{generated_dir}/*/*/*.png")

        if not image_files:
            warnings.warn("No generated images found for VQA evaluation")
            return np.array([]), []

        # Prepare dataset for VQA evaluation
        dataset = []
        for img_file in image_files:
            prompt = self._path_to_prompt(img_file)
            dataset.append(
                {
                    "images": [img_file],
                    "texts": [prompt],
                    "metadata": {"path": img_file, "prompt": prompt},
                }
            )

        if self.config.verbose:
            print(f"Evaluating {len(dataset)} image-text pairs...")

        # Evaluate in batches
        scores = self.model.batch_forward(
            dataset=dataset, batch_size=self.config.vqa_batch_size
        )
        scores_array = scores.cpu().numpy()

        if self.config.verbose:
            print(
                f"VQA Score: {scores_array.mean():.3f} ± {scores_array.std():.3f} ({len(scores_array)} samples)"
            )

        return scores_array, dataset

    def cleanup(self):
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()


class EvaluationSuite:
    """Main evaluation suite combining multiple metrics."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.dino_evaluator = DINOv2Evaluator(self.config)
        self.vqa_evaluator = VQAEvaluator(self.config)

    def evaluate(
        self,
        generated_dir: str,
        reference_dir: str,
        instances: List[str],
        mask_dir: Optional[str] = None,
        metrics: List[str] = ["dino", "vqa"],
    ) -> Dict[str, np.ndarray]:
        """Run complete evaluation suite."""

        if self.config.verbose:
            print(f"Starting evaluation with metrics: {metrics}")
            print(f"Generated images directory: {generated_dir}")
            print(f"Reference images directory: {reference_dir}")
            print(f"Evaluating {len(instances)} instances: {instances}")

        results = {}

        # DINOv2 evaluation
        if "dino" in metrics:
            try:
                dino_scores, dino_instance_scores = self.dino_evaluator.evaluate(
                    generated_dir, reference_dir, instances, mask_dir
                )
                results["dino"] = dino_scores
                results["dino_by_instance"] = dino_instance_scores
            except Exception as e:
                warnings.warn(f"DINOv2 evaluation failed: {e}")
                results["dino"] = np.array([])
                results["dino_by_instance"] = {}
            finally:
                self.dino_evaluator.cleanup()

        # VQA evaluation
        if "vqa" in metrics:
            try:
                vqa_scores, vqa_dataset = self.vqa_evaluator.evaluate(generated_dir)
                results["vqa"] = vqa_scores
                results["vqa_dataset"] = vqa_dataset
            except Exception as e:
                warnings.warn(f"VQA evaluation failed: {e}")
                results["vqa"] = np.array([])
                results["vqa_dataset"] = []
            finally:
                self.vqa_evaluator.cleanup()

        return results

    def save_results(self, results: Dict[str, np.ndarray], output_file: str):
        """Save evaluation results to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare summary statistics
        summary = {}
        for metric_name, scores in results.items():
            if isinstance(scores, np.ndarray) and len(scores) > 0:
                summary[metric_name] = {
                    "mean": float(scores.mean()),
                    "std": float(scores.std()),
                    "count": int(len(scores)),
                    "min": float(scores.min()),
                    "max": float(scores.max()),
                }

        # Save summary as JSON
        summary_file = output_path.with_suffix(".json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        if self.config.verbose:
            print(f"Results saved to {summary_file}")

        # Optionally save individual scores
        if self.config.save_individual_scores:
            scores_file = output_path.with_suffix(".npz")
            # Filter out non-array data for npz saving
            arrays_to_save = {
                k: v for k, v in results.items() if isinstance(v, np.ndarray)
            }
            np.savez(scores_file, **arrays_to_save)

            if self.config.verbose:
                print(f"Individual scores saved to {scores_file}")

    def cleanup(self):
        """Clean up all evaluators."""
        self.dino_evaluator.cleanup()
        self.vqa_evaluator.cleanup()
