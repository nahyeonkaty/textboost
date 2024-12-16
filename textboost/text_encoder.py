from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask, _prepare_4d_attention_mask)
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.models.clip.modeling_clip import (CLIPEncoder,
                                                    CLIPTextConfig,
                                                    CLIPTextEmbeddings,
                                                    CLIPTextModel,
                                                    CLIPTextTransformer)


class TextBoostModel(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        null_embed = torch.zeros(
            config.max_position_embeddings,
            config.hidden_size,
            dtype=torch.float,
        )
        self.register_buffer("null_embedding", null_embed)
        self._use_fixed_special_embedding = False

    def set_null_embedding(self, null_embedding):
        if isinstance(null_embedding, str):
            null_embedding = torch.load(null_embedding)
        self.null_embedding = null_embedding
        self._use_fixed_special_embedding = True

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        null_pos = (input_ids[:, 1] == 49407)
        if null_pos.any():
            # print(output[0][null_pos].shape)  # [N, 77, 1024]
            # print(self.null_embedding.shape)
            output[0][null_pos] = (
                self.null_embedding
                .unsqueeze(0)
                .repeat(output[0][null_pos].shape[0], 1, 1)
            )

        if self._use_fixed_special_embedding:
            output[0][:, 0] = (
                self.null_embedding[0]
                .unsqueeze(0)
                .repeat(output[0].shape[0], 1)
            )
        return output
