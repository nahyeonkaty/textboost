from typing import Any, Iterable

import torch
import torch.nn.functional as F

from ..ti_utils import forced_weight_norm
from .base import BaseTrainer


class SDTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        accelerator,
        train_dataloader,
        max_train_steps: int,
        progress_bar=None,
        callbacks=None,
        unet=None,
        text_encoder=None,
        vae=None,
        noise_scheduler=None,
        emb_optimizer=None,
        optimizer=None,
        emb_lr_scheduler=None,
        lr_scheduler=None,
        args=None,
        weight_dtype: torch.dtype = torch.float32,
        tokenizer=None,
        placeholder_token_ids=None,
        orig_embeds_params=None,
        active_expand_bank=None,
        accumulate_models: Iterable[Any] | None = None,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            train_dataloader=train_dataloader,
            max_train_steps=max_train_steps,
            progress_bar=progress_bar,
            callbacks=callbacks,
        )
        self.unet = unet
        self.text_encoder = text_encoder
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.emb_optimizer = emb_optimizer
        self.optimizer = optimizer
        self.emb_lr_scheduler = emb_lr_scheduler
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.weight_dtype = weight_dtype
        self.tokenizer = tokenizer
        self.placeholder_token_ids = placeholder_token_ids
        self.orig_embeds_params = orig_embeds_params
        self.active_expand_bank = active_expand_bank
        self._accumulate_models = (
            tuple(accumulate_models)
            if accumulate_models is not None
            else (self.unet, self.text_encoder)
        )

    @property
    def accumulate_models(self) -> Iterable[Any]:
        return self._accumulate_models

    def prepare_batch(self, batch: Any) -> Any:
        prepared = {
            "pixel_values": batch["pixel_values"].to(
                self.accelerator.device, dtype=self.vae.dtype
            ),
            "input_ids": batch["input_ids"].to(self.accelerator.device),
            "adapter_mask": batch["adapter_mask"].squeeze(1),
            "batch": batch,
        }
        if self.args.regularization > 0.0:
            prepared["reg_input_ids"] = batch["reg_input_ids"].to(
                self.accelerator.device
            )
        return prepared

    def encode_conditioning(self, prepared_batch: Any) -> Any:
        pixel_values = prepared_batch["pixel_values"]
        input_ids = prepared_batch["input_ids"]
        adapter_mask = prepared_batch["adapter_mask"]

        with torch.no_grad():
            model_input = self.vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * self.vae.config.scaling_factor

        noise = torch.randn_like(model_input)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (model_input.shape[0],),
            device=model_input.device,
        )
        noisy_model_input = self.noise_scheduler.add_noise(
            model_input, noise, timesteps
        )

        text_outputs = self.text_encoder(
            input_ids,
            attention_mask=None,
            adapter_mask=adapter_mask,
            expand_adapter_mask=adapter_mask.to(self.accelerator.device),
        )
        encoder_hidden_states = text_outputs.last_hidden_state
        expand_hidden_states = text_outputs.expand_hidden_state

        cross_attention_kwargs = {
            "adapter_mask": adapter_mask.to(self.accelerator.device)
        }
        if expand_hidden_states is not None:
            cross_attention_kwargs["expand_hidden_states"] = {
                key: value.to(dtype=self.weight_dtype)
                for key, value in expand_hidden_states.items()
            }
        model_pred = self.unet(
            noisy_model_input.to(self.weight_dtype),
            timesteps,
            encoder_hidden_states.to(self.weight_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        return {
            "model_pred": model_pred,
            "target": target,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def compute_loss(
        self,
        prepared_batch: Any,
        conditioning: Any,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        batch = prepared_batch["batch"]
        input_ids = prepared_batch["input_ids"]
        adapter_mask = prepared_batch["adapter_mask"]
        encoder_hidden_states = conditioning["encoder_hidden_states"]

        diffusion_loss = F.mse_loss(
            conditioning["model_pred"].float(),
            conditioning["target"].float(),
            reduction="none",
        )
        if "mask" in batch:
            mask = batch["mask"].to(self.accelerator.device, dtype=self.weight_dtype)
            diffusion_loss = (
                (diffusion_loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])
            ).mean()
        else:
            diffusion_loss = diffusion_loss.mean()
        loss = diffusion_loss

        if self.args.regularization > 0.0:
            with torch.no_grad():
                reg_encoder_hidden_states = self.text_encoder(
                    prepared_batch["reg_input_ids"],
                    attention_mask=None,
                    return_dict=False,
                    adapter_mask=adapter_mask,
                )[0]
            mask = (input_ids < 49407).float()
            reg_loss = 1.0 - F.cosine_similarity(
                x1=encoder_hidden_states,
                x2=reg_encoder_hidden_states,
                dim=-1,
            )
            reg_loss = mask * reg_loss
            loss = loss + self.args.regularization * reg_loss.mean()

        return loss, {"loss": float(diffusion_loss.detach().item())}

    def optimizer_step(
        self,
        *,
        loss: torch.Tensor,
        prepared_batch: Any,
        conditioning: Any,
        step: int,
    ) -> dict[str, float] | None:
        del prepared_batch
        del conditioning
        del step

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            params_to_clip = list(
                self.accelerator.unwrap_model(
                    self.text_encoder
                ).text_model.encoder.parameters()
            )
            if self.active_expand_bank is not None:
                params_to_clip += list(self.active_expand_bank.parameters())
            self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)

        self.emb_optimizer.step()
        self.optimizer.step()
        self.emb_lr_scheduler.step()
        self.lr_scheduler.step()
        self.emb_optimizer.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

        index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
        index_no_updates[
            min(self.placeholder_token_ids) : max(self.placeholder_token_ids) + 1
        ] = False
        with torch.no_grad():
            self.accelerator.unwrap_model(
                self.text_encoder
            ).get_input_embeddings().weight[index_no_updates] = self.orig_embeds_params[
                index_no_updates
            ]

        norm = forced_weight_norm(
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            index=self.placeholder_token_ids,
            magnitude=self.args.max_embedding_norm,
        )
        return {"v_norm": float(norm.mean().item())}

    def build_logs(
        self,
        *,
        step: int,
        loss_logs: dict[str, float] | None,
        optimizer_logs: dict[str, float] | None,
    ) -> dict[str, float]:
        del step
        logs: dict[str, float] = {}
        if loss_logs:
            logs.update(loss_logs)
        logs["lr_emb"] = self.emb_lr_scheduler.get_last_lr()[0]
        logs["lr"] = self.lr_scheduler.get_last_lr()[0]
        if optimizer_logs:
            logs.update(optimizer_logs)
        return logs
