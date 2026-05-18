import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import (
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    retrieve_timesteps,
    rescale_noise_cfg,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    is_torch_xla_available,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TextBoostPipeline(StableDiffusionPipeline):
    def make_adapter_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        adapter_mask = attention_mask.clone()
        for i, ids in enumerate(input_ids):
            if ids.max() < 49408:
                adapter_mask[i] = 0.0
            else:
                known_tokens = (ids < 49408).long()
                unk_idx = torch.argmin(known_tokens)
                adapter_mask[i, :unk_idx] = 0.0
        adapter_mask = adapter_mask.unsqueeze(-1)
        return adapter_mask

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        self._last_prompt_expand_hidden_states = None
        self._last_negative_expand_hidden_states = None
        active_expand_bank = (
            self.text_encoder.get_expand_adapter_bank()
            if hasattr(self.text_encoder, "get_expand_adapter_bank")
            else None
        )
        prompt_expand_hidden_states = None
        negative_expand_hidden_states = None
        text_adapter_mask: torch.Tensor | None = None
        uncond_adapter_mask: torch.Tensor | None = None

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids
            text_adapter_mask = self.make_adapter_mask(
                text_input_ids,
                text_inputs.attention_mask,
            ).to(device)

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_outputs = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                    adapter_mask=text_adapter_mask,
                    expand_adapter_bank=active_expand_bank,
                    expand_adapter_mask=text_adapter_mask,
                )
                prompt_embeds = prompt_outputs.last_hidden_state
            else:
                prompt_outputs = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    adapter_mask=text_adapter_mask,
                    expand_adapter_bank=active_expand_bank,
                    expand_adapter_mask=text_adapter_mask,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(
                    prompt_embeds
                )
            if active_expand_bank is not None:
                prompt_expand_hidden_states = prompt_outputs.expand_hidden_state

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        def _repeat_expand_hidden_states(
            expand_hidden_states: dict[str, torch.Tensor] | None,
        ) -> dict[str, torch.Tensor] | None:
            if expand_hidden_states is None:
                return None
            repeated: dict[str, torch.Tensor] = {}
            for key, value in expand_hidden_states.items():
                value = value.to(dtype=prompt_embeds_dtype, device=device)
                bsz_local, seq_len_local, _ = value.shape
                value = value.repeat(1, num_images_per_prompt, 1)
                value = value.view(
                    bsz_local * num_images_per_prompt,
                    seq_len_local,
                    -1,
                )
                repeated[key] = value
            return repeated

        bs_embed, seq_len, _ = prompt_embeds.shape
        if text_adapter_mask is None:
            text_adapter_mask = torch.ones(
                (bs_embed, seq_len, 1),
                device=device,
                dtype=prompt_embeds_dtype,
            )
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        text_adapter_mask = text_adapter_mask.repeat(1, num_images_per_prompt, 1)
        text_adapter_mask = text_adapter_mask.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        prompt_expand_hidden_states = _repeat_expand_hidden_states(
            prompt_expand_hidden_states
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_adapter_mask = self.make_adapter_mask(
                uncond_input.input_ids,
                uncond_input.attention_mask,
            ).to(device)

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_outputs = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
                adapter_mask=uncond_adapter_mask,
                expand_adapter_bank=active_expand_bank,
                expand_adapter_mask=uncond_adapter_mask,
            )
            negative_prompt_embeds = negative_outputs.last_hidden_state
            if active_expand_bank is not None:
                negative_expand_hidden_states = negative_outputs.expand_hidden_state

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            if uncond_adapter_mask is None:
                uncond_adapter_mask = torch.ones(
                    (
                        negative_prompt_embeds.shape[0],
                        seq_len,
                        1,
                    ),
                    device=device,
                    dtype=negative_prompt_embeds.dtype,
                )

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            uncond_adapter_mask = uncond_adapter_mask.repeat(
                1, num_images_per_prompt, 1
            )
            uncond_adapter_mask = uncond_adapter_mask.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            negative_expand_hidden_states = _repeat_expand_hidden_states(
                negative_expand_hidden_states
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        adapter_masks = (
            [uncond_adapter_mask, text_adapter_mask]
            if do_classifier_free_guidance
            else [text_adapter_mask]
        )
        self._last_prompt_expand_hidden_states = prompt_expand_hidden_states
        self._last_negative_expand_hidden_states = negative_expand_hidden_states
        return prompt_embeds, negative_prompt_embeds, adapter_masks

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = (
                height * self.vae_scale_factor,
                width * self.vae_scale_factor,
            )
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds, adapter_masks = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        prompt_expand_hidden_states = getattr(
            self, "_last_prompt_expand_hidden_states", None
        )
        negative_expand_hidden_states = getattr(
            self, "_last_negative_expand_hidden_states", None
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            adapter_masks = torch.cat(adapter_masks)
        expand_hidden_states = prompt_expand_hidden_states
        if self.do_classifier_free_guidance and prompt_expand_hidden_states is not None:
            if negative_expand_hidden_states is None:
                negative_expand_hidden_states = {
                    key: torch.zeros_like(value)
                    for key, value in prompt_expand_hidden_states.items()
                }
            expand_hidden_states = {
                key: torch.cat(
                    [
                        negative_expand_hidden_states[key],
                        prompt_expand_hidden_states[key],
                    ],
                    dim=0,
                )
                for key in prompt_expand_hidden_states
            }

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 6.3 Optionally get Adapter Mask.
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {"adapter_mask": adapter_masks}
        else:
            cross_attention_kwargs["adapter_mask"] = adapter_masks
        if expand_hidden_states is not None:
            cross_attention_kwargs["expand_hidden_states"] = expand_hidden_states
        if self.cross_attention_kwargs is not None:
            cross_attention_kwargs.update(self.cross_attention_kwargs)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    # cross_attention_kwargs=self.cross_attention_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    @classmethod
    def load_sd_pipeline(
        cls,
        model,
        checkpoint_path,
    ):
        sdxl = "stable-diffusion-xl" in model
        print(f"SDXL: {sdxl}")
        file_list = os.listdir(checkpoint_path)

        unet = UNet2DConditionModel.from_pretrained(
            model,
            subfolder="unet",
        )
        unet.set_attn_processor(TextBoostAttnProcessor())

        text_encoder = ClipTextModel.from_pretrained(
            model,
            subfolder="text_encoder",
        )
        if "text_encoder" in file_list:
            text_encoder_path = os.path.join(checkpoint_path, "text_encoder")
            text_encoder.load_adapter(text_encoder_path)
            text_encoder.set_adapter_mask()  # TODO: remove this line.
            text_encoder.replace_lora_forward()

        if sdxl:
            text_encoder_2 = TextModelWithProjection.from_pretrained(
                model,
                subfolder="text_encoder_2",
            )
            text_encoder_2_path = os.path.join(checkpoint_path, "text_encoder_2")
            text_encoder_2.load_adapter(text_encoder_2_path)
            text_encoder_2.set_adapter_mask()
            text_encoder_2.replace_lora_forward()
            pipeline = TextBoostSDXLPipeline.from_pretrained(
                model,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                unet=unet,
                safety_checker=None,
            )
        else:
            pipeline = cls.from_pretrained(
                model,
                text_encoder=text_encoder,
                unet=unet,
                safety_checker=None,
            )

        # 2. Optionally load U-Net LoRA weights or Adapter.
        if "pytorch_lora_weights.safetensors" in file_list:
            pipeline.load_lora_weights(
                checkpoint_path,
                weight_name="pytorch_lora_weights.safetensors",
            )
            print("Loaded LoRA weights from checkpoint.")
        elif "adapter.bin" in file_list:
            adapter_state_dict = torch.load(
                os.path.join(checkpoint_path, "adapter.bin")
            )
            for k, v in adapter_state_dict.items():
                if "adapter.up" in k:
                    dim = v.shape[0]
                    rank = v.shape[1]
                    if k.replace("weight", "bias") in adapter_state_dict:
                        bias = True
                    else:
                        bias = False
                    # print(k, v.shape, bias)
                    break
            attn_to_ks = []
            attn_to_vs = []
            for name, module in unet.named_modules():
                if "attn2.to_k" in name:
                    attn_to_ks.append(module)
                elif "attn2.to_v" in name:
                    attn_to_vs.append(module)
            for m in attn_to_ks:
                adapter = Adapter(dim, rank, bias)
                setattr(m, "adapter", adapter)
            for m in attn_to_vs:
                adapter = Adapter(dim, rank, bias)
                setattr(m, "adapter", adapter)
            unet.load_state_dict(adapter_state_dict, strict=False)
            print("Loaded Adapter from checkpoint.")

        # 3. Load learned embeddings.
        emb_dict = torch.load(os.path.join(checkpoint_path, "learned_embeds.bin"))
        identifiers = []
        for key, value in emb_dict.items():
            identifier = load_new_token(
                text_encoder,
                pipeline.tokenizer,
                placeholder_token=key,
                learned_embedding=value,
            )
            identifiers.append(identifier)
        if sdxl:
            emb_dict = torch.load(os.path.join(checkpoint_path, "learned_embeds_2.bin"))
            for key, value in emb_dict.items():
                identifier = load_new_token(
                    text_encoder_2,
                    pipeline.tokenizer_2,
                    placeholder_token=key,
                    learned_embedding=value,
                )
                identifiers.append(identifier)
        print("Loaded learned embeddings from checkpoint.")
        print(identifiers)
        return pipeline, identifiers
