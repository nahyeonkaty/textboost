from typing import Any, Iterable

import torch
import torch.nn.functional as F

from ..text_encoders.clip import build_adapter_mask_from_inputs
from ..ti_utils import forced_weight_norm
from .base import BaseTrainer


class SDXLTrainer(BaseTrainer):
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
        text_encoder_2=None,
        vae=None,
        noise_scheduler=None,
        emb_optimizer=None,
        optimizer=None,
        emb_lr_scheduler=None,
        lr_scheduler=None,
        args=None,
        weight_dtype: torch.dtype = torch.float32,
        tokenizer=None,
        tokenizer_2=None,
        added_token_ids=None,
        added_token_ids_2=None,
        orig_embeds_params=None,
        orig_embeds_params_2=None,
        enable_unet_adapter: bool = False,
        active_expand_bank=None,
        expand_param_ids: set[int] | None = None,
        ti_reference_mode: bool = False,
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
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.emb_optimizer = emb_optimizer
        self.optimizer = optimizer
        self.emb_lr_scheduler = emb_lr_scheduler
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.weight_dtype = weight_dtype
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.added_token_ids = added_token_ids
        self.added_token_ids_2 = added_token_ids_2
        self.orig_embeds_params = orig_embeds_params
        self.orig_embeds_params_2 = orig_embeds_params_2
        self.enable_unet_adapter = enable_unet_adapter
        self.active_expand_bank = active_expand_bank
        self.expand_param_ids = (
            expand_param_ids if expand_param_ids is not None else set()
        )
        self.ti_reference_mode = ti_reference_mode
        self._accumulate_models = (
            tuple(accumulate_models)
            if accumulate_models is not None
            else (self.unet, self.text_encoder, self.text_encoder_2)
        )

    @property
    def accumulate_models(self) -> Iterable[Any]:
        return self._accumulate_models

    def prepare_batch(self, batch: Any) -> Any:
        input_ids = batch["input_ids"].to(self.accelerator.device).squeeze(1)
        attention_mask = batch["attention_mask"].to(self.accelerator.device).squeeze(1)
        input_ids_2 = batch["input_ids_2"].to(self.accelerator.device).squeeze(1)
        attention_mask_2 = (
            batch["attention_mask_2"].to(self.accelerator.device).squeeze(1)
        )
        if self.enable_unet_adapter:
            adapter_mask = build_adapter_mask_from_inputs(
                input_ids,
                attention_mask,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            adapter_mask = None
        return {
            "batch": batch,
            "pixel_values": batch["image"].to(
                self.accelerator.device, dtype=self.vae.dtype
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_ids_2": input_ids_2,
            "attention_mask_2": attention_mask_2,
            "adapter_mask": adapter_mask,
        }

    def encode_conditioning(self, prepared_batch: Any) -> Any:
        pixel_values = prepared_batch["pixel_values"]
        input_ids = prepared_batch["input_ids"]
        attention_mask = prepared_batch["attention_mask"]
        input_ids_2 = prepared_batch["input_ids_2"]
        attention_mask_2 = prepared_batch["attention_mask_2"]
        adapter_mask = prepared_batch["adapter_mask"]
        batch = prepared_batch["batch"]

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        if self.args.offset_noise:
            noise = torch.randn_like(latents) + 0.1 * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=latents.device
            )
        else:
            noise = torch.randn_like(latents)

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states_1 = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-2]
        encoder_output_2 = self.text_encoder_2(
            input_ids_2,
            attention_mask=attention_mask_2,
            output_hidden_states=True,
            expand_adapter_mask=(
                adapter_mask.to(self.accelerator.device)
                if adapter_mask is not None
                else None
            ),
            expand_prefix_hidden_states=encoder_hidden_states_1,
        )
        encoder_hidden_states_2 = encoder_output_2.hidden_states[-2]
        expand_hidden_states = encoder_output_2.expand_hidden_state

        original_size = [
            (
                batch["original_size"][0][i].item(),
                batch["original_size"][1][i].item(),
            )
            for i in range(bsz)
        ]
        crop_top_left = [
            (
                batch["crop_top_left"][0][i].item(),
                batch["crop_top_left"][1][i].item(),
            )
            for i in range(bsz)
        ]
        target_size = (self.args.resolution, self.args.resolution)
        add_time_ids = torch.cat(
            [
                torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                for i in range(bsz)
            ]
        ).to(self.accelerator.device, dtype=self.weight_dtype)
        added_cond_kwargs = {
            "text_embeds": encoder_output_2[0],
            "time_ids": add_time_ids,
        }
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states_1, encoder_hidden_states_2], dim=-1
        ).to(dtype=self.weight_dtype)

        cross_attention_kwargs = None
        if adapter_mask is not None:
            cross_attention_kwargs = {
                "adapter_mask": adapter_mask.to(self.accelerator.device)
            }
            if expand_hidden_states is not None:
                cross_attention_kwargs["expand_hidden_states"] = {
                    key: value.to(dtype=self.weight_dtype)
                    for key, value in expand_hidden_states.items()
                }

        model_pred = self.unet(
            noisy_latents.to(dtype=self.weight_dtype),
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        return {"model_pred": model_pred, "target": target}

    def compute_loss(
        self,
        prepared_batch: Any,
        conditioning: Any,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        del prepared_batch
        diffusion_loss = F.mse_loss(
            conditioning["model_pred"].float(),
            conditioning["target"].float(),
            reduction="none",
        )
        diffusion_loss = diffusion_loss.mean()
        return diffusion_loss, {"loss": float(diffusion_loss.detach().item())}

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
        if self.accelerator.sync_gradients and self.args.max_grad_norm > 0.0:
            params_to_clip = self.accelerator.unwrap_model(
                self.text_encoder
            ).text_model.encoder.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
            params_to_clip = self.accelerator.unwrap_model(
                self.text_encoder_2
            ).text_model.encoder.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
            if self.active_expand_bank is not None:
                self.accelerator.clip_grad_norm_(
                    list(self.active_expand_bank.parameters()),
                    self.args.max_grad_norm,
                )
            if self.args.unet_lora_rank > 0:
                params_to_clip = [
                    p
                    for p in self.accelerator.unwrap_model(self.unet).parameters()
                    if id(p) not in self.expand_param_ids
                ]
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.args.max_grad_norm
                )

        self.emb_optimizer.step()
        self.optimizer.step()
        self.emb_lr_scheduler.step()
        self.lr_scheduler.step()
        self.emb_optimizer.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

        index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
        index_no_updates[min(self.added_token_ids) : max(self.added_token_ids) + 1] = (
            False
        )
        index_no_updates_2 = torch.ones((len(self.tokenizer_2),), dtype=torch.bool)
        index_no_updates_2[
            min(self.added_token_ids_2) : max(self.added_token_ids_2) + 1
        ] = False

        with torch.no_grad():
            self.accelerator.unwrap_model(
                self.text_encoder
            ).get_input_embeddings().weight[index_no_updates] = self.orig_embeds_params[
                index_no_updates
            ]
            self.accelerator.unwrap_model(
                self.text_encoder_2
            ).get_input_embeddings().weight[
                index_no_updates_2
            ] = self.orig_embeds_params_2[index_no_updates_2]

        if self.ti_reference_mode:
            return {}

        v_norm = forced_weight_norm(
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            index=self.added_token_ids,
            magnitude=self.args.max_embedding_norm,
        )
        v_norm_2 = forced_weight_norm(
            text_encoder=self.accelerator.unwrap_model(self.text_encoder_2),
            index=self.added_token_ids_2,
            magnitude=self.args.max_embedding_norm_2,
        )
        return {
            "v_norm": float(v_norm.mean().item()),
            "v_norm_2": float(v_norm_2.mean().item()),
        }

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
        logs["lr"] = self.lr_scheduler.get_last_lr()[0]
        logs["lr_emb"] = self.emb_lr_scheduler.get_last_lr()[0]
        if optimizer_logs:
            logs.update(optimizer_logs)
        return logs
