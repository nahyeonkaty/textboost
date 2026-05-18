from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch


class BaseTrainer(ABC):
    def __init__(
        self,
        *,
        accelerator,
        train_dataloader,
        max_train_steps: int,
        progress_bar=None,
        callbacks=None,
    ) -> None:
        self.accelerator = accelerator
        self.train_dataloader = train_dataloader
        self.max_train_steps = max_train_steps
        self.progress_bar = progress_bar
        self.callbacks = callbacks

    @property
    @abstractmethod
    def accumulate_models(self) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare_batch(self, batch: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def encode_conditioning(self, prepared_batch: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self,
        prepared_batch: Any,
        conditioning: Any,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        raise NotImplementedError

    @abstractmethod
    def optimizer_step(
        self,
        *,
        loss: torch.Tensor,
        prepared_batch: Any,
        conditioning: Any,
        step: int,
    ) -> dict[str, float] | None:
        raise NotImplementedError

    def build_logs(
        self,
        *,
        step: int,
        loss_logs: dict[str, float] | None,
        optimizer_logs: dict[str, float] | None,
    ) -> dict[str, float]:
        logs: dict[str, float] = {}
        if loss_logs:
            logs.update(loss_logs)
        if optimizer_logs:
            logs.update(optimizer_logs)
        return logs

    def on_sync_step_end(self, *, step: int) -> None:
        return

    def on_step_end(self, *, step: int, logs: dict[str, float]) -> None:
        return

    def run(self, *, initial_step: int = 0) -> int:
        if self.callbacks is not None:
            self.callbacks.on_train_start(initial_step=initial_step)

        step = initial_step
        train_iterator = iter(self.train_dataloader)
        accumulate_models = list(self.accumulate_models)

        while step < self.max_train_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
            prepared_batch = self.prepare_batch(batch)

            with self.accelerator.accumulate(*accumulate_models):
                conditioning = self.encode_conditioning(prepared_batch)
                loss, loss_logs = self.compute_loss(prepared_batch, conditioning)
                optimizer_logs = self.optimizer_step(
                    loss=loss,
                    prepared_batch=prepared_batch,
                    conditioning=conditioning,
                    step=step,
                )

            logs = self.build_logs(
                step=step,
                loss_logs=loss_logs,
                optimizer_logs=optimizer_logs,
            )
            self.on_step_end(step=step, logs=logs)
            if self.callbacks is not None:
                self.callbacks.on_step_end(step=step, logs=logs)

            if self.accelerator.sync_gradients:
                if self.progress_bar is not None:
                    self.progress_bar.update(1)
                step += 1
                self.on_sync_step_end(step=step)
                if self.callbacks is not None:
                    self.callbacks.on_sync_step_end(step=step)

        if self.callbacks is not None:
            self.callbacks.on_train_end(final_step=step)
        return step
