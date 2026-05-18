"""
Unified evaluation runner that combines generation and metrics evaluation.

This module provides high-level interfaces for running complete evaluation
workflows, integrating both the planning/generation and metrics computation.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .core import (
    EvaluationPlanner,
    EvaluationPlan,
    print_plan_summary,
    print_batch_summary,
)

# Try to import metrics evaluation if available
try:
    from .metrics import EvaluationSuite, EvaluationConfig

    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    EvaluationSuite = None
    EvaluationConfig = None


class UnifiedEvaluationRunner:
    """Unified runner for evaluation workflows."""

    def __init__(
        self,
        seeds: List[int] = None,
        model_type: str = "sana",
        batch_size: int = 4,
        device: str = "cuda",
        quiet: bool = False,
    ):
        self.planner = EvaluationPlanner(seeds=seeds, model_type=model_type)
        self.batch_size = batch_size
        self.device = device
        self.quiet = quiet

    def run_planning_mode(
        self,
        method_dirs: List[str],
        prompts_file: Optional[str] = None,
        prompts_list: Optional[List[str]] = None,
        instances: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        dry_run: bool = False,
    ) -> Union[EvaluationPlan, List[EvaluationPlan]]:
        """Run planning mode - create evaluation plans without generation."""

        if len(method_dirs) == 1:
            # Single method planning
            plan = self.planner.create_plan(
                method_dirs[0], prompts_file, prompts_list, instances
            )

            if not self.quiet:
                print_plan_summary(plan, "planning")

            if not dry_run:
                output_path = output_dir or str(Path(method_dirs[0]) / "evaluation")
                plan_file = plan.save_plan(output_path)
                if not self.quiet:
                    print(f"Plan saved to: {plan_file}")

            return plan

        else:
            # Batch planning
            plans = self.planner.create_batch_plan(
                method_dirs,
                prompts_file=prompts_file,
                prompts_list=prompts_list,
                instances=instances,
            )

            if not self.quiet:
                print_batch_summary(plans)

            if not dry_run and output_dir:
                batch_data = {
                    "plans": [plan.to_dict() for plan in plans],
                    "summary": {
                        "total_methods": len(plans),
                        "total_images": sum(plan.total_images for plan in plans),
                    },
                }

                batch_file = Path(output_dir) / "batch_plans.json"
                batch_file.parent.mkdir(parents=True, exist_ok=True)
                with open(batch_file, "w") as f:
                    json.dump(batch_data, f, indent=2)

                if not self.quiet:
                    print(f"Batch plans saved to: {batch_file}")

            return plans

    def run_simple_mode(
        self,
        method_dirs: List[str],
        prompts_file: Optional[str] = None,
        prompts_list: Optional[List[str]] = None,
        instances: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Run simple mode - create directory structure and file organization."""

        if len(method_dirs) == 1:
            # Single method
            plan = self.planner.create_plan(
                method_dirs[0], prompts_file, prompts_list, instances
            )

            output_path = output_dir or str(Path(method_dirs[0]) / "evaluation")
            files_to_generate = plan.create_directory_structure(output_path, overwrite)
            plan_file = plan.save_plan(output_path, include_file_structure=True)

            result = {
                "success": True,
                "plan": plan.to_dict(),
                "files_to_generate": files_to_generate,
                "output_dir": output_path,
                "plan_file": plan_file,
                "files_count": sum(len(files) for files in files_to_generate.values()),
            }

            if not self.quiet:
                print_plan_summary(plan, "simple")
                print(f"Files to generate: {result['files_count']}")
                print(f"Output directory: {output_path}")

            return result

        else:
            # Batch mode
            results = []
            for method_dir in method_dirs:
                try:
                    single_result = self.run_simple_mode(
                        [method_dir],
                        prompts_file,
                        prompts_list,
                        instances,
                        None,
                        overwrite,
                    )
                    single_result["method"] = os.path.basename(method_dir)
                    results.append(single_result)
                except Exception as e:
                    results.append(
                        {
                            "success": False,
                            "method": os.path.basename(method_dir),
                            "error": str(e),
                        }
                    )

            # Print batch summary
            if not self.quiet:
                successful = [r for r in results if r["success"]]
                failed = [r for r in results if not r["success"]]

                print(f"\n{'=' * 80}")
                print("BATCH SIMPLE MODE SUMMARY")
                print(f"{'=' * 80}")
                print(f"Total methods: {len(results)}")
                print(f"Successful: {len(successful)}")
                print(f"Failed: {len(failed)}")

                if successful:
                    total_files = sum(r["files_count"] for r in successful)
                    print(f"Total files to generate: {total_files}")

                if failed:
                    print("Failed methods:")
                    for r in failed:
                        print(f"  - {r['method']}: {r['error']}")
                print(f"{'=' * 80}")

            return {"batch_results": results}

    def run_metrics_evaluation(
        self,
        generated_dir: str,
        reference_dir: str,
        instances: Optional[List[str]] = None,
        mask_dir: Optional[str] = None,
        metrics: List[str] = None,
    ) -> Dict[str, Any]:
        """Run metrics evaluation on generated images."""

        if not HAS_METRICS:
            raise ImportError(
                "Metrics evaluation requires textboost.evaluation.metrics module"
            )

        # Determine metrics
        if not metrics:
            metrics = ["dino", "vqa"]
        elif "all" in metrics:
            metrics = ["dino", "vqa"]

        # Create evaluation config
        config = EvaluationConfig(
            device=self.device, batch_size=self.batch_size, verbose=not self.quiet
        )

        # Create evaluator and run
        evaluator = EvaluationSuite(config)

        try:
            results = evaluator.evaluate(
                generated_dir=generated_dir,
                reference_dir=reference_dir,
                instances=instances,
                mask_dir=mask_dir,
                metrics=metrics,
            )

            return {"success": True, "results": results, "error": None}

        except Exception as e:
            return {"success": False, "results": None, "error": str(e)}

        finally:
            evaluator.cleanup()

    def run_full_evaluation(
        self,
        method_dir: str,
        prompts_file: Optional[str] = None,
        prompts_list: Optional[List[str]] = None,
        instances: Optional[List[str]] = None,
        reference_dir: str = "datasets/dreambooth",
        mask_dir: Optional[str] = None,
        metrics: List[str] = None,
        output_dir: Optional[str] = None,
        overwrite: bool = False,
        skip_generation: bool = False,
        skip_evaluation: bool = False,
    ) -> Dict[str, Any]:
        """Run full evaluation workflow."""

        results = {
            "method_dir": method_dir,
            "method_name": os.path.basename(method_dir),
            "generation": None,
            "evaluation": None,
            "success": False,
        }

        output_path = output_dir or str(Path(method_dir) / "evaluation")

        try:
            # Step 1: Generation planning and setup
            if not skip_generation:
                if not self.quiet:
                    print("Setting up generation structure...")

                gen_result = self.run_simple_mode(
                    [method_dir],
                    prompts_file,
                    prompts_list,
                    instances,
                    output_path,
                    overwrite,
                )
                results["generation"] = gen_result

                if not self.quiet:
                    print("Generation structure created.")
                    print("TODO: Implement actual image generation here.")

            # Step 2: Metrics evaluation
            if not skip_evaluation:
                if not self.quiet:
                    print("Running metrics evaluation...")

                eval_result = self.run_metrics_evaluation(
                    generated_dir=output_path,
                    reference_dir=reference_dir,
                    instances=instances,
                    mask_dir=mask_dir,
                    metrics=metrics,
                )
                results["evaluation"] = eval_result

                if eval_result["success"]:
                    # Save combined results
                    combined_results = {
                        "method_info": {
                            "method_dir": method_dir,
                            "method_name": os.path.basename(method_dir),
                            "output_dir": output_path,
                        },
                        "generation_info": results["generation"],
                        "evaluation_results": eval_result["results"],
                    }

                    results_file = Path(output_path) / "full_evaluation_results.json"
                    with open(results_file, "w") as f:
                        json.dump(combined_results, f, indent=2)

                    if not self.quiet:
                        print(f"Results saved to: {results_file}")

                else:
                    if not self.quiet:
                        print(f"Evaluation failed: {eval_result['error']}")

            results["success"] = True

        except Exception as e:
            results["error"] = str(e)
            if not self.quiet:
                print(f"Full evaluation failed: {e}")

        return results

    def run_legacy_evaluation(
        self,
        generated_dir: str,
        reference_dir: str,
        instances: Optional[List[str]] = None,
        mask_dir: Optional[str] = None,
        metrics: List[str] = None,
        results_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run evaluation using the legacy evaluate.py script."""

        # Build command for legacy script
        cmd = ["python", "scripts/evaluate.py", generated_dir, reference_dir]

        if instances:
            cmd.extend(["--instances"] + instances)

        if metrics and "all" not in metrics:
            cmd.extend(["--metrics"] + metrics)

        if mask_dir:
            cmd.extend(["--mask-dir", mask_dir])

        if results_file:
            cmd.extend(["--output", results_file])

        cmd.extend(["--device", self.device, "--batch-size", str(self.batch_size)])

        if self.quiet:
            cmd.append("--quiet")

        try:
            if not self.quiet:
                print(f"Running legacy evaluation: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            return {
                "success": True,
                "output": result.stdout,
                "error": None,
                "command": cmd,
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "output": e.stdout if e.stdout else "",
                "error": e.stderr if e.stderr else str(e),
                "command": cmd,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": "Legacy evaluate.py script not found",
                "command": cmd,
            }


# Convenience function for quick evaluation
def quick_evaluate(method_dir: str, mode: str = "simple", **kwargs) -> Dict[str, Any]:
    """Quick evaluation function with sensible defaults."""
    runner = UnifiedEvaluationRunner()

    if mode == "simple":
        return runner.run_simple_mode([method_dir], **kwargs)
    elif mode == "plan":
        kwargs.setdefault("dry_run", True)
        return runner.run_planning_mode([method_dir], **kwargs)
    elif mode == "full":
        return runner.run_full_evaluation(method_dir, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


__all__ = ["UnifiedEvaluationRunner", "quick_evaluate", "HAS_METRICS"]
