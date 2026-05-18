from typing import Any, Iterable

import numpy as np
import torch
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from ..ti_utils import forced_weight_norm
from .base import BaseTrainer


class SanaTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        accelerator,
        train_dataloader,
        max_train_steps: int,
        progress_bar=None,
        callbacks=None,
        dit=None,
        text_encoder=None,
        vae=None,
        text_encoding_pipeline=None,
        noise_scheduler_copy=None,
        emb_optimizer=None,
        optimizer=None,
        emb_lr_scheduler=None,
        lr_scheduler=None,
        args=None,
        dit_dtype: torch.dtype = torch.float32,
        tokenizer=None,
        added_token_ids=None,
        orig_embeds_params=None,
        use_attention_kwargs: bool = False,
        accumulate_models: Iterable[Any] | None = None,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            train_dataloader=train_dataloader,
            max_train_steps=max_train_steps,
            progress_bar=progress_bar,
            callbacks=callbacks,
        )
        self.dit = dit
        self.text_encoder = text_encoder
        self.vae = vae
        self.text_encoding_pipeline = text_encoding_pipeline
        self.noise_scheduler_copy = noise_scheduler_copy
        self.emb_optimizer = emb_optimizer
        self.optimizer = optimizer
        self.emb_lr_scheduler = emb_lr_scheduler
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.dit_dtype = dit_dtype
        self.tokenizer = tokenizer
        self.added_token_ids = added_token_ids
        self.orig_embeds_params = orig_embeds_params
        self.use_attention_kwargs = use_attention_kwargs
        self._accumulate_models = (
            tuple(accumulate_models)
            if accumulate_models is not None
            else (self.dit, self.text_encoder)
        )

    @property
    def accumulate_models(self) -> Iterable[Any]:
        return self._accumulate_models

    def _compute_text_embeddings(self, prompt):
        self.text_encoding_pipeline = self.text_encoding_pipeline.to(
            self.accelerator.device
        )
        if np.random.rand() < self.args.chi_prob:
            complex_human_instruction = self.args.complex_human_instruction
        else:
            complex_human_instruction = None
        try:
            outputs = self.text_encoding_pipeline._get_gemma_prompt_embeds(
                prompt,
                device=self.accelerator.device,
                dtype=self.accelerator.unwrap_model(self.dit).dtype,
                max_sequence_length=self.args.max_sequence_length,
                complex_human_instruction=complex_human_instruction,
                output_adapter_mask=True,
                output_expand_hidden_states=True,
            )
        except TypeError:
            outputs = self.text_encoding_pipeline._get_gemma_prompt_embeds(
                prompt,
                device=self.accelerator.device,
                dtype=self.accelerator.unwrap_model(self.dit).dtype,
                max_sequence_length=self.args.max_sequence_length,
                complex_human_instruction=complex_human_instruction,
                output_adapter_mask=True,
            )
        if len(outputs) >= 4:
            prompt_embeds, prompt_attention_mask, adapter_mask, expand_hidden_states = (
                outputs[0],
                outputs[1],
                outputs[2],
                outputs[3],
            )
        elif len(outputs) == 3:
            prompt_embeds, prompt_attention_mask, adapter_mask = outputs
            expand_hidden_states = None
        else:
            prompt_embeds, prompt_attention_mask = outputs[:2]
            adapter_mask = None
            expand_hidden_states = None
        return prompt_embeds, prompt_attention_mask, adapter_mask, expand_hidden_states

    def _get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler_copy.sigmas.to(
            device=self.accelerator.device, dtype=dtype
        )
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(
            self.accelerator.device
        )
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def prepare_batch(self, batch: Any) -> Any:
        return {
            "prompt": batch["prompt"],
            "pixel_values": batch["pixel_values"].to(
                self.accelerator.device, dtype=self.vae.dtype
            ),
        }

    def encode_conditioning(self, prepared_batch: Any) -> Any:
        prompt = prepared_batch["prompt"]
        pixel_values = prepared_batch["pixel_values"]

        model_input = self.vae.encode(pixel_values).latent
        model_input = model_input * self.vae.config.scaling_factor

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.args.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.args.logit_mean,
            logit_std=self.args.logit_std,
            mode_scale=self.args.mode_scale,
        )
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(
            device=model_input.device
        )

        sigmas = self._get_sigmas(
            timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
        )
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        (
            prompt_embeds,
            prompt_attention_mask,
            adapter_mask,
            expand_hidden_states,
        ) = self._compute_text_embeddings(prompt)

        attention_kwargs = None
        if self.use_attention_kwargs:
            attention_kwargs = {}
            if adapter_mask is not None:
                attention_kwargs["adapter_mask"] = adapter_mask.to(
                    self.accelerator.device
                )
            if expand_hidden_states is not None:
                attention_kwargs["expand_hidden_states"] = {
                    key: value.to(dtype=self.dit_dtype)
                    for key, value in expand_hidden_states.items()
                }
            if not attention_kwargs:
                attention_kwargs = None

        model_pred = self.dit(
            hidden_states=noisy_model_input.to(dtype=self.dit_dtype),
            encoder_hidden_states=prompt_embeds.to(dtype=self.dit_dtype),
            timestep=timesteps,
            encoder_attention_mask=prompt_attention_mask,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.args.weighting_scheme,
            sigmas=sigmas,
        )
        target = noise - model_input
        return {"model_pred": model_pred, "target": target, "weighting": weighting}

    def compute_loss(
        self,
        prepared_batch: Any,
        conditioning: Any,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        del prepared_batch
        loss = torch.mean(
            (
                conditioning["weighting"].float()
                * (conditioning["model_pred"].float() - conditioning["target"].float())
                ** 2
            ).reshape(conditioning["target"].shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss, {"loss": float(loss.detach().item())}

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
            params_to_clip = [
                param
                for name, param in self.accelerator.unwrap_model(
                    self.text_encoder
                ).named_parameters()
                if param.requires_grad and "embed_tokens" not in name
            ]
            if params_to_clip:
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
        with torch.no_grad():
            self.accelerator.unwrap_model(
                self.text_encoder
            ).get_input_embeddings().weight[index_no_updates] = self.orig_embeds_params[
                index_no_updates
            ]

        norm = forced_weight_norm(
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            index=self.added_token_ids,
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
