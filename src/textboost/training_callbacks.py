from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Sequence

from diffusers.utils import make_image_grid


class TrainingCallback:
    def on_train_start(self, *, initial_step: int) -> None:
        return

    def on_step_end(self, *, step: int, logs: dict[str, float]) -> None:
        return

    def on_sync_step_end(self, *, step: int) -> None:
        return

    def on_train_end(self, *, final_step: int) -> None:
        return


class CallbackHandler:
    def __init__(self, callbacks: Sequence[TrainingCallback]) -> None:
        self.callbacks = list(callbacks)

    def on_train_start(self, *, initial_step: int) -> None:
        for callback in self.callbacks:
            callback.on_train_start(initial_step=initial_step)

    def on_step_end(self, *, step: int, logs: dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_step_end(step=step, logs=logs)

    def on_sync_step_end(self, *, step: int) -> None:
        for callback in self.callbacks:
            callback.on_sync_step_end(step=step)

    def on_train_end(self, *, final_step: int) -> None:
        for callback in self.callbacks:
            callback.on_train_end(final_step=final_step)


class MetricWriterCallback(TrainingCallback):
    def __init__(self, accelerator, progress_bar) -> None:
        self.accelerator = accelerator
        self.progress_bar = progress_bar

    def on_step_end(self, *, step: int, logs: dict[str, float]) -> None:
        self.progress_bar.set_postfix(**logs)
        self.accelerator.log(logs, step=step)


class CheckpointCallback(TrainingCallback):
    def __init__(
        self,
        *,
        accelerator,
        output_dir: str | Path,
        checkpointing_steps: int,
        checkpoints_total_limit: int | None,
        logger,
        save_artifact_fn: Callable[[str, int], None],
    ) -> None:
        self.accelerator = accelerator
        self.output_dir = str(output_dir)
        self.checkpointing_steps = checkpointing_steps
        self.checkpoints_total_limit = checkpoints_total_limit
        self.logger = logger
        self.save_artifact_fn = save_artifact_fn

    def _prune_checkpoints(self) -> None:
        if self.checkpoints_total_limit is None:
            return

        checkpoints = os.listdir(self.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) < self.checkpoints_total_limit:
            return

        num_to_remove = len(checkpoints) - self.checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[:num_to_remove]
        self.logger.info(
            "%s checkpoints already exist, removing %s checkpoints",
            len(checkpoints),
            len(removing_checkpoints),
        )
        self.logger.info("removing checkpoints: %s", ", ".join(removing_checkpoints))
        for removing_checkpoint in removing_checkpoints:
            shutil.rmtree(os.path.join(self.output_dir, removing_checkpoint))

    def on_sync_step_end(self, *, step: int) -> None:
        if not self.accelerator.is_main_process:
            return
        if self.checkpointing_steps <= 0 or step % self.checkpointing_steps != 0:
            return

        self._prune_checkpoints()
        save_path = os.path.join(self.output_dir, f"checkpoint-{step}")
        self.accelerator.save_state(save_path)
        self.save_artifact_fn(save_path, step)
        self.logger.info("Saved state to %s", save_path)


class ValidationSamplerCallback(TrainingCallback):
    def __init__(
        self,
        *,
        accelerator,
        validation_steps: int,
        validation_prompts: Sequence[str] | None,
        num_validation_images: int,
        output_dir: str | Path,
        run_validation_fn: Callable[[int], list],
        run_initial_validation: bool = True,
    ) -> None:
        self.accelerator = accelerator
        self.validation_steps = validation_steps
        self.validation_prompts = list(validation_prompts or [])
        self.num_validation_images = num_validation_images
        self.output_dir = str(output_dir)
        self.run_validation_fn = run_validation_fn
        self.run_initial_validation = run_initial_validation

    def _run_validation(self, step: int) -> None:
        if not self.accelerator.is_main_process:
            return
        if not self.validation_prompts:
            return

        images = self.run_validation_fn(step)
        if not images:
            return

        rows = len(self.validation_prompts)
        cols = self.num_validation_images
        image_grid = make_image_grid(images, rows, cols)
        image_grid.save(os.path.join(self.output_dir, f"validation_{step}.jpg"))

    def on_train_start(self, *, initial_step: int) -> None:
        if self.run_initial_validation and initial_step == 0:
            self._run_validation(0)

    def on_sync_step_end(self, *, step: int) -> None:
        if self.validation_steps <= 0:
            return
        if step % self.validation_steps == 0:
            self._run_validation(step)


class PooledEmbeddingTrackerCallback(TrainingCallback):
    def __init__(
        self,
        *,
        accelerator,
        enabled: bool,
        log_steps: int,
        initialize_fn: Callable[[], dict],
        collect_with_adapter_fn: Callable[[dict], object],
        collect_token_only_fn: Callable[[dict], object],
        summarize_fn: Callable[[dict, object, object, int], dict[str, float]],
    ) -> None:
        self.accelerator = accelerator
        self.enabled = enabled
        self.log_steps = log_steps
        self.initialize_fn = initialize_fn
        self.collect_with_adapter_fn = collect_with_adapter_fn
        self.collect_token_only_fn = collect_token_only_fn
        self.summarize_fn = summarize_fn
        self.tracker: dict | None = None

    def _log_once(self, step: int) -> None:
        if self.tracker is None:
            return
        pooled_with_adapter = self.collect_with_adapter_fn(self.tracker)
        pooled_token_only = self.collect_token_only_fn(self.tracker)
        summary = self.summarize_fn(
            self.tracker, pooled_with_adapter, pooled_token_only, step
        )
        self.accelerator.log(summary, step=step)

    def on_train_start(self, *, initial_step: int) -> None:
        if not self.enabled or not self.accelerator.is_main_process:
            return
        self.tracker = self.initialize_fn()
        self._log_once(initial_step)

    def on_sync_step_end(self, *, step: int) -> None:
        if not self.enabled or not self.accelerator.is_main_process:
            return
        if self.log_steps <= 0:
            return
        if step % self.log_steps == 0:
            self._log_once(step)
