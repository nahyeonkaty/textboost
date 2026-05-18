import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import SanaTransformer2DModel
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import SanaLoraLoaderMixin
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import (
    ASPECT_RATIO_2048_BIN,
)
from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
from diffusers.pipelines.sana.pipeline_sana import (
    SanaPipeline,
    retrieve_timesteps,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    scale_lora_layers,
    unscale_lora_layers,
    logging,
)

from textboost.attention_processor import (
    SanaAttnProcessor,
    build_textboost_attn_processors,
)
from textboost.expand_bank import (
    build_expand_bank_from_state_dict,
    iter_cross_attention_to_k_layers,
)
from textboost.ti_utils import load_new_token
from textboost.adapters import Adapter, TrfConfig, attach_adapters_to_model
from textboost.text_encoders.gemma2 import TextModel as Gemma2TextModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TextBoostSanaPipeline(SanaPipeline):
    def make_adapter_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        if causal:
            adapter_mask = attention_mask.clone()
            for i, ids in enumerate(input_ids):
                # input_ids: seq_len
                # print("ids", ids, ids.max(), ids.shape)
                if ids.max() < 256000:
                    # print("null mask", adapter_mask)
                    adapter_mask[i] = 0.0
                else:
                    known_tokens = (ids < 256000).long()
                    unk_idx = torch.argmin(known_tokens)
                    # print("known tokens", known_tokens)
                    # print("mask", adapter_mask[i].shape, unk_idx)
                    adapter_mask[i, :unk_idx] = 0.0
                    # print("adapter_mask", adapter_mask)
        else:
            adapter_mask = input_ids >= 256000
        adapter_mask = adapter_mask.unsqueeze(-1)
        return adapter_mask

    def _get_gemma_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        dtype: torch.dtype,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
        output_input_ids: bool = False,
        output_adapter_mask: bool = False,
        output_expand_hidden_states: bool = False,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            clean_caption (`bool`, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            max_sequence_length (`int`, defaults to 300): Maximum sequence length to use for the prompt.
            complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`):
                If `complex_human_instruction` is not empty, the function will use the complex Human instruction for
                the prompt.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)

        # prepare complex human instruction
        if not complex_human_instruction:
            max_length_all = max_sequence_length
        else:
            if isinstance(complex_human_instruction, str):
                chi_prompt = complex_human_instruction
            else:
                chi_prompt = "\n".join(complex_human_instruction)
            prompt = [chi_prompt + p for p in prompt]
            num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
            max_length_all = num_chi_prompt_tokens + max_sequence_length - 2

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)

        adapter_mask = self.make_adapter_mask(
            text_input_ids,
            prompt_attention_mask,
            causal=False,
        )
        # print('---')
        # print(prompt)
        # print(text_input_ids.shape, adapter_mask.shape)
        # print(text_input_ids[0].view(-1))
        # print(adapter_mask[0].view(-1))
        # print('---')

        active_expand_bank = None
        if output_expand_hidden_states and hasattr(
            self.text_encoder, "get_expand_adapter_bank"
        ):
            active_expand_bank = self.text_encoder.get_expand_adapter_bank()

        prompt_outputs = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=prompt_attention_mask,
            adapter_mask=adapter_mask.to(device),
            expand_adapter_bank=active_expand_bank,
            expand_adapter_mask=(
                adapter_mask.to(device) if active_expand_bank is not None else None
            ),
        )
        prompt_embeds = prompt_outputs[0].to(dtype=dtype)
        expand_hidden_states = (
            prompt_outputs.expand_hidden_state
            if active_expand_bank is not None
            else None
        )

        outputs = (prompt_embeds, prompt_attention_mask)
        if output_input_ids:
            outputs += (text_input_ids,)
        if output_adapter_mask:
            outputs += (adapter_mask,)
        if output_expand_hidden_states:
            outputs += (expand_hidden_states,)
        return outputs

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
        lora_scale: Optional[float] = None,
        output_adapter_mask: bool = False,
        output_expand_hidden_states: bool = False,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                PixArt-Alpha, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Sana, it's should be the embeddings of the "" string.
            clean_caption (`bool`, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            max_sequence_length (`int`, defaults to 300): Maximum sequence length to use for the prompt.
            complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`):
                If `complex_human_instruction` is not empty, the function will use the complex Human instruction for
                the prompt.
        """

        if device is None:
            device = self._execution_device

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        else:
            dtype = None

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SanaLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        # See Section 3.1. of the paper.
        max_length = max_sequence_length
        select_index = [0] + list(range(-max_length + 1, 0))
        active_expand_bank = None
        if output_expand_hidden_states and hasattr(
            self.text_encoder, "get_expand_adapter_bank"
        ):
            active_expand_bank = self.text_encoder.get_expand_adapter_bank()
        prompt_expand_hidden_states = None
        negative_expand_hidden_states = None
        adapter_mask = None
        negative_adapter_mask = None

        def _slice_expand_hidden_states(
            expand_hidden_states: Dict[str, torch.Tensor] | None,
            index: list[int],
        ) -> Dict[str, torch.Tensor] | None:
            if expand_hidden_states is None:
                return None
            sliced: Dict[str, torch.Tensor] = {}
            for key, value in expand_hidden_states.items():
                if value is None:
                    continue
                sliced[key] = value[:, index]
            return sliced if sliced else None

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)

            # prepare complex human instruction
            if not complex_human_instruction:
                max_length_all = max_length
            else:
                if isinstance(complex_human_instruction, str):
                    chi_prompt = complex_human_instruction
                else:
                    chi_prompt = "\n".join(complex_human_instruction)
                prompt = [chi_prompt + p for p in prompt]
                num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
                max_length_all = num_chi_prompt_tokens + max_length - 2

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length_all,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(device)

            adapter_mask = self.make_adapter_mask(text_input_ids, prompt_attention_mask)
            # print(prompt)
            # print(text_input_ids.view(-1))
            # print(adapter_mask.view(-1))

            prompt_outputs = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_attention_mask,
                adapter_mask=adapter_mask.to(device),
                expand_adapter_bank=active_expand_bank,
                expand_adapter_mask=(
                    adapter_mask.to(device) if active_expand_bank is not None else None
                ),
            )
            prompt_embeds = prompt_outputs[0][:, select_index]
            if active_expand_bank is not None:
                prompt_expand_hidden_states = _slice_expand_hidden_states(
                    prompt_outputs.expand_hidden_state,
                    select_index,
                )
            prompt_attention_mask = prompt_attention_mask[:, select_index]
        else:
            adapter_mask = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        def _repeat_expand_hidden_states(
            expand_hidden_states: Dict[str, torch.Tensor] | None,
            repeat_count: int,
        ) -> Dict[str, torch.Tensor] | None:
            if expand_hidden_states is None:
                return None
            repeated: Dict[str, torch.Tensor] = {}
            for key, value in expand_hidden_states.items():
                if value is None:
                    continue
                if value.shape[0] == 0:
                    repeated[key] = value
                    continue
                repeated[key] = value.repeat_interleave(repeat_count, dim=0)
            return repeated if repeated else None

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        prompt_attention_mask = prompt_attention_mask.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        if adapter_mask is not None:
            adapter_mask = adapter_mask.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_expand_hidden_states = _repeat_expand_hidden_states(
            prompt_expand_hidden_states,
            num_images_per_prompt,
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = (
                [negative_prompt] * batch_size
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            negative_outputs = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                device=device,
                dtype=dtype,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
                complex_human_instruction=False,
                output_adapter_mask=True,
                output_expand_hidden_states=output_expand_hidden_states,
            )
            if output_expand_hidden_states:
                (
                    negative_prompt_embeds,
                    negative_prompt_attention_mask,
                    negative_adapter_mask,
                    negative_expand_hidden_states,
                ) = negative_outputs
            else:
                (
                    negative_prompt_embeds,
                    negative_prompt_attention_mask,
                    negative_adapter_mask,
                ) = negative_outputs

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            negative_prompt_attention_mask = (
                negative_prompt_attention_mask.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
            )
            if negative_adapter_mask is not None:
                negative_adapter_mask = negative_adapter_mask.repeat_interleave(
                    num_images_per_prompt,
                    dim=0,
                )
            negative_expand_hidden_states = _repeat_expand_hidden_states(
                negative_expand_hidden_states,
                num_images_per_prompt,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None
            negative_adapter_mask = None

        if self.text_encoder is not None:
            if isinstance(self, SanaLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if output_adapter_mask and output_expand_hidden_states:
            return (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                prompt_expand_hidden_states,
                negative_expand_hidden_states,
                adapter_mask,
                negative_adapter_mask,
            )
        if output_adapter_mask:
            return (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                adapter_mask,
                negative_adapter_mask,
            )
        if output_expand_hidden_states:
            return (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                prompt_expand_hidden_states,
                negative_expand_hidden_states,
            )
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: int = 1024,
        width: int = 1024,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
        # complex_human_instruction: List[str] = [
        #     "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
        #     "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        #     "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        #     "Here are examples of how to transform or refine prompts:",
        #     "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        #     "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        #     "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        #     "User Prompt: ",
        # ],
    ) -> Union[SanaPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 20):
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
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            attention_kwargs:
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `300`):
                Maximum sequence length to use with the `prompt`.
            complex_human_instruction (`List[str]`, *optional*):
                Instructions for complex human attention:
                https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

        Examples:

        Returns:
            [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        if use_resolution_binning:
            if self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_2048_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 16:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(
                height, width, ratios=aspect_ratio_bin
            )

        self.check_inputs(
            prompt,
            height,
            width,
            callback_on_step_end_tensor_inputs,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = (
            self.attention_kwargs.get("scale", None)
            if self.attention_kwargs is not None
            else None
        )

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            prompt_expand_hidden_states,
            negative_expand_hidden_states,
            adapter_mask,
            negative_adapter_mask,
        ) = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
            lora_scale=lora_scale,
            output_adapter_mask=True,
            output_expand_hidden_states=True,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

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
                        negative_expand_hidden_states.get(
                            key, torch.zeros_like(prompt_expand_hidden_states[key])
                        ),
                        prompt_expand_hidden_states[key],
                    ],
                    dim=0,
                )
                for key in prompt_expand_hidden_states
            }

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Create adapter mask.
        if self.do_classifier_free_guidance:
            if negative_adapter_mask is None and adapter_mask is not None:
                negative_adapter_mask = torch.zeros_like(adapter_mask)
            if adapter_mask is None and negative_adapter_mask is not None:
                adapter_mask = torch.zeros_like(negative_adapter_mask)
            if negative_adapter_mask is not None and adapter_mask is not None:
                full_adapter_mask = torch.cat(
                    [
                        negative_adapter_mask.to(device),
                        adapter_mask.to(device),
                    ],
                    dim=0,
                )
            else:
                full_adapter_mask = None
        else:
            full_adapter_mask = (
                adapter_mask.to(device) if adapter_mask is not None else None
            )
        inference_attention_kwargs = (
            dict(self.attention_kwargs) if self.attention_kwargs is not None else None
        )
        if expand_hidden_states is not None:
            if inference_attention_kwargs is None:
                inference_attention_kwargs = {}
            inference_attention_kwargs["expand_hidden_states"] = expand_hidden_states

        for k, attn_proc in self.transformer.attn_processors.items():
            if isinstance(attn_proc, SanaAttnProcessor):
                attn_proc.set_adapter_mask(full_adapter_mask)

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timestep,
                    return_dict=False,
                    attention_kwargs=inference_attention_kwargs,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
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

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(
                    image, orig_width, orig_height
                )

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return SanaPipelineOutput(images=image)

    @classmethod
    def from_checkpoint(
        cls,
        model: str,
        checkpoint_path: str,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple["TextBoostSanaPipeline", List[str]]:
        """
        Load a TextBoostSanaPipeline from a checkpoint.

        Args:
            model (`str`):
                The name of the model to load.
            checkpoint_path (`str`):
                The path to the checkpoint directory.

        Returns:
            `TextBoostSanaPipeline`: The loaded pipeline.
        """
        checkpoint_dir = Path(checkpoint_path)
        file_list = (
            {path.name for path in checkpoint_dir.iterdir()}
            if checkpoint_dir.exists()
            else set()
        )

        dit = SanaTransformer2DModel.from_pretrained(
            model,
            subfolder="transformer",
        )

        text_encoder = Gemma2TextModel.from_pretrained(
            model,
            subfolder="text_encoder",
        )
        config_path = None
        for candidate in (
            checkpoint_dir / "config.json",
            checkpoint_dir / "text_encoder" / "config.json",
            checkpoint_dir / "adapter" / "text_encoder" / "config.json",
        ):
            if candidate.exists():
                config_path = candidate
                break
        if config_path is not None:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            text_encoder = attach_adapters_to_model(
                text_encoder,
                TrfConfig(**config_data),
            )
        elif "text_encoder" in file_list:
            text_encoder_dir = checkpoint_dir / "text_encoder"
            if (
                text_encoder_dir.exists()
                and (text_encoder_dir / "adapter_config.json").exists()
            ):
                text_encoder.load_adapter(str(text_encoder_dir))
                text_encoder.set_adapter_mask()
                text_encoder.replace_lora_forward()

        text_encoder_state_path = None
        for candidate in (
            checkpoint_dir / "text_encoder.bin",
            checkpoint_dir / "text_encoder" / "text_encoder.bin",
            checkpoint_dir / "adapter" / "text_encoder" / "text_encoder.bin",
        ):
            if candidate.exists():
                text_encoder_state_path = candidate
                break
        if text_encoder_state_path is not None:
            state_dict = torch.load(text_encoder_state_path, map_location="cpu")
            text_encoder.load_state_dict(state_dict, strict=False)

        pipeline = cls.from_pretrained(
            model,
            text_encoder=text_encoder,
            transformer=dit,
        )
        if dtype is not None:
            pipeline.to(dtype=dtype)
            pipeline.vae.to(dtype=torch.float32)

        # 2. Optionally load U-Net LoRA weights or Adapter.
        # if "transformer" in file_list:
        #     pipeline.load_lora_weights(
        #         os.path.join(checkpoint_path, "transformer"),
        #         weight_name="diffusion_pytorch_model.safetensors",
        #     )
        #     print("Loaded LoRA weights from checkpoint.")
        if "pytorch_lora_weights.safetensors" in file_list:
            pipeline.load_lora_weights(
                checkpoint_path,
                weight_name="pytorch_lora_weights.safetensors",
            )
            print("Loaded LoRA weights from checkpoint.")

        expand_bank_path = None
        for candidate in (
            checkpoint_dir / "expand_bank.bin",
            checkpoint_dir / "text_encoder" / "expand_bank.bin",
            checkpoint_dir / "adapter" / "text_encoder" / "expand_bank.bin",
            checkpoint_dir / "adapter" / "unet" / "expand_bank.bin",
        ):
            if candidate.exists():
                expand_bank_path = candidate
                break

        if expand_bank_path is not None:
            expand_state_dict = torch.load(expand_bank_path, map_location="cpu")
            expand_bank = build_expand_bank_from_state_dict(expand_state_dict)
            text_encoder.set_expand_adapter_bank(expand_bank)
            pipeline.transformer.set_attn_processor(
                build_textboost_attn_processors(pipeline.transformer, use_sana=True)
            )
            available_cross_layers = list(
                iter_cross_attention_to_k_layers(pipeline.transformer)
            )
            print(
                f"Loaded expand bank from checkpoint. "
                f"adapters={len(expand_bank.adapters)}, "
                f"cross_attn_layers={len(available_cross_layers)}"
            )
        else:
            legacy_adapter_path = None
            for candidate in (
                checkpoint_dir / "adapter.bin",
                checkpoint_dir / "adapter" / "unet" / "adapter.bin",
            ):
                if candidate.exists():
                    legacy_adapter_path = candidate
                    break
        if expand_bank_path is None and legacy_adapter_path is not None:
            adapter_state_dict = torch.load(
                legacy_adapter_path,
                map_location="cpu",
            )
            dim = None
            rank = None
            bias = False
            for k, v in adapter_state_dict.items():
                if "adapter.up" in k:
                    dim = v.shape[0]
                    rank = v.shape[1]
                    bias = k.replace("weight", "bias") in adapter_state_dict
                    break
            if dim is None or rank is None:
                raise ValueError(
                    "Could not infer legacy SANA adapter shape from checkpoint."
                )
            attn_to_ks = []
            for name, module in dit.named_modules():
                if "attn2.to_k" in name:
                    attn_to_ks.append(module)
            for m in attn_to_ks:
                adapter = Adapter(dim, rank, bias)
                setattr(m, "adapter", adapter)
            mis, une = dit.load_state_dict(adapter_state_dict, strict=False)
            # if len(mis) > 0:
            #     print(f"Missing keys in adapter state dict: {mis}")
            if len(une) > 0:
                print(f"Unexpected keys in adapter state dict: {une}")
            pipeline.transformer.set_attn_processor(
                build_textboost_attn_processors(pipeline.transformer, use_sana=True)
            )
            print("Loaded Adapter from checkpoint.")

        # 3. Load learned embeddings.
        embeds_path = checkpoint_dir / "learned_embeds.bin"
        emb_dict = torch.load(embeds_path)
        identifiers = []
        for key, value in emb_dict.items():
            identifier = load_new_token(
                text_encoder,
                pipeline.tokenizer,
                placeholder=key,
                learned_embedding=value,
                joiner="",
            )
            identifiers.append(identifier)
        print("Loaded learned embeddings from checkpoint.")
        print(identifiers)
        pipeline.identifiers = identifiers
        return pipeline, identifiers
