from __future__ import annotations

from dataclasses import dataclass

import torch
from safetensors.torch import save_file


@dataclass
class NewToken:
    placeholder: str
    identifier: str
    initializer: str
    token_ids: list[int]
    num_vectors: int
    concept_identifier: str | None = None

    def dict(self) -> dict:
        return {
            "placeholder": self.placeholder,
            "identifier": self.identifier,
            "initializer": self.initializer,
            "token_id": self.token_ids,
            "num_vectors": self.num_vectors,
        }


def add_new_token(
    tokenizer,
    text_encoder,
    placeholder: str,  # User input: "<concept>".
    init_token: torch.Tensor | str | None = None,
    pad_style: str = "tb",
    joiner: str = "",
) -> NewToken:
    # Convert the initializer_token, placeholder_token to ids.
    if init_token is not None:
        initializer_token_ids = tokenizer.encode(
            init_token,
            add_special_tokens=False,
        )
        num_vectors = len(initializer_token_ids)
    else:
        initializer_token_ids = [None]
        num_vectors = 1

    # Add the placeholder token in tokenizer
    placeholder_tokens = [placeholder]

    # add dummy tokens for multi-vector
    additional_tokens = []
    if num_vectors > 1:
        if placeholder.endswith(">") and pad_style == "tb":
            placeholder_tokens[0] = placeholder[:-1] + "_0>"
            for i in range(1, num_vectors):
                additional_tokens.append(f"{placeholder[:-1]}_{i}>")
        else:
            for i in range(1, num_vectors):
                additional_tokens.append(f"{placeholder}_{i}")
    placeholder_tokens += additional_tokens
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    token_identifier = joiner.join(placeholder_tokens)
    if num_added_tokens != num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    if len(initializer_token_ids) != len(placeholder_tokens):
        raise ValueError(
            f"Number of tokens in the initializer_token and placeholder_token should be the same. "
            f"initializer_token: {init_token}, placeholder_token: {placeholder}"
        )

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for token_id, initializer_token_id in zip(
        placeholder_token_ids, initializer_token_ids
    ):
        if initializer_token_id is not None:
            token_embed = token_embeds[initializer_token_id].detach().clone()
            token_embeds[token_id] = token_embed

    new_token = NewToken(
        placeholder=placeholder,
        identifier=token_identifier,
        initializer=init_token if isinstance(init_token, str) else "",
        token_ids=placeholder_token_ids,
        num_vectors=num_vectors,
    )
    return new_token


def save_embeddings(
    text_encoder,
    new_tokens: list[NewToken],
    save_path: str,
    safe_serialization: bool = True,
) -> None:
    learned_embeds_dict = {}
    for new_token in new_tokens:
        key = new_token.placeholder
        token_ids = new_token.token_ids
        learned_embeds = text_encoder.get_input_embeddings().weight[
            min(token_ids) : max(token_ids) + 1
        ]
        learned_embeds_dict[key] = learned_embeds.detach().cpu()

    if safe_serialization:
        save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def load_new_token(
    text_encoder,
    tokenizer,
    placeholder: str,
    learned_embedding: torch.Tensor,
    joiner: str = "",
) -> str:
    num_vectors = learned_embedding.shape[0]
    # Add the placeholder token in tokenizer
    placeholder_tokens = [placeholder]
    if num_vectors > 1:
        if placeholder.endswith(">"):
            placeholder_tokens[0] = placeholder[:-1] + "_0>"
            for i in range(1, num_vectors):
                placeholder_tokens.append(f"{placeholder[:-1]}_{i}>")
        else:
            for i in range(1, num_vectors):
                placeholder_tokens.append(f"{placeholder}_{i}")
    identifier = joiner.join(placeholder_tokens)
    tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))

    token_embeds = text_encoder.get_input_embeddings().weight.data
    dtype = token_embeds.dtype
    device = token_embeds.device
    learned_embedding = learned_embedding.to(device, dtype=dtype)
    for i, token_id in enumerate(placeholder_token_ids):
        token_embed = learned_embedding[i]
        token_embeds[token_id] = token_embed
    return identifier


@torch.no_grad()
def forced_weight_norm(
    text_encoder, index: int | list[int], magnitude: float = 1.0
) -> torch.Tensor:
    embeddings = text_encoder.get_input_embeddings()
    target_embeddings = embeddings.weight[index]
    norm = torch.norm(target_embeddings, dim=-1, keepdim=True)
    scale = magnitude / norm
    scale[scale > 1.0] = 1.0
    new_embeddings = scale * target_embeddings
    text_encoder.get_input_embeddings().weight[index] = new_embeddings
    return norm
