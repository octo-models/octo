# Written by Dibya
from dataclasses import asdict, dataclass, replace
import logging

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.components.transformer import Transformer
from orca.utils.typing import Dict, PRNGKey, Sequence, Union


@dataclass
class PrefixGroup:
    """A group of tokens that will be at the beginning of the token sequence."""

    name: str
    tokens: jnp.ndarray  # with shape (batch, n_tokens, token_embedding_size)
    attends_to: Sequence[str]  # other prefix groups this will attend to
    attends_to_self: bool = True  # whether this group attends to itself

    def __post_init__(self):
        assert self.tokens.ndim == 3


@dataclass
class TimestepGroup:
    """A group of tokens that is repeated for each timestep."""

    name: str
    tokens: jnp.ndarray  # with shape (batch, horizon, n_tokens, token_embedding_size)
    attends_to: Sequence[str]  # other groups this will attend to with <= timestep
    attends_to_past: Sequence[str] = tuple()  # attend to with < timestep. Rarely used.
    attends_to_self: bool = True  # whether this group attends to itself

    def __post_init__(self):
        assert self.tokens.ndim == 4


@dataclass
class TokenMetadata:
    """Useful metadata for computing attention masks"""

    name: str
    timestep: int  # -1 for prefix tokens
    attends_to: Sequence[str]
    attends_to_past: Sequence[str] = tuple()
    attends_to_self: bool = True

    @classmethod
    def create(cls, group: Union[PrefixGroup, TimestepGroup], timestep: int):
        group_dict = asdict(group)
        group_dict.pop("tokens")
        return cls(
            timestep=timestep,
            **group_dict,
        )

    def should_attend_to(self, other_metadata: "TokenMetadata") -> bool:
        if other_metadata.name in self.attends_to:
            return self.timestep >= other_metadata.timestep
        elif other_metadata.name in self.attends_to_past:
            return self.timestep > other_metadata.timestep
        elif self.name == other_metadata.name:
            if self.timestep == other_metadata.timestep:
                return self.attends_to_self
            return False  # Otherwise, should have been caught by attends_to or attends_to_past
        else:
            return False


def split_tokens(ary, n_tokens_per_group, axis):
    cumsum = np.cumsum(n_tokens_per_group)
    return jnp.split(ary, cumsum, axis=axis)


class BlockTransformer(nn.Module):
    num_layers: int = 4
    mlp_dim: int = 1024
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        prefix_groups: Sequence[PrefixGroup],
        timestep_groups: Sequence[TimestepGroup],
        timestep_pad_mask: jnp.ndarray,
        train: bool,
        verbose: bool = False,
    ):
        """
        Args:
            prefix_groups: A list of PrefixGroup objects.
                Each group has tokens with shape (batch, n_tokens, token_embedding_size)
                Each group also dictates which other groups it will attend to.
            timestep_groups: A list of TimestepGroup objects.
                Each group has tokens with shape (batch, horizon, n_tokens, token_embedding_size)
                Each group also dictates which other groups it will attend to.
            timestep_pad_mask: A boolean mask of shape (batch, horizon) indicating which timesteps are padding.
            train: Whether to use dropout.

        Returns:
            embedding_dict: A dictionary {
                    "task": task_embeddings, # shape (batch, tokens_per_task, token_embedding_size)
                    "obs": obs_embeddings, # shape (batch, horizon, tokens_per_obs, token_embedding_size)
                    **{k: embedding for k in computation_groups} # shape (batch, horizon, computation_groups[k].shape[-2], token_embedding_size)
                }

        Note: Horizon can be anything <= max_horizon.
        """
        if verbose:
            self.pretty_print_attention_mask(prefix_groups, timestep_groups)

        horizon = timestep_groups[0].tokens.shape[1]
        assert all([group.tokens.shape[1] == horizon for group in timestep_groups])

        attention_mask = self.generate_attention_mask(
            prefix_groups, timestep_groups, timestep_pad_mask
        )
        self.sow("intermediates", "attention_mask", attention_mask)

        input_tokens = self.assemble_input_tokens(prefix_groups, timestep_groups)
        self.sow("intermediates", "input_tokens", input_tokens)

        transformer = Transformer(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_attention_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            add_position_embedding=False,  # we add our own
        )
        output = transformer(input_tokens, attention_mask, train=train)

        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        n_prefix_tokens = sum(tokens_per_prefix_group)

        prefix_embeddings, timestep_embeddings = jnp.split(
            output, [n_prefix_tokens], axis=1
        )
        if len(prefix_groups) > 0:
            prefix_embeddings_split = split_tokens(
                prefix_embeddings, tokens_per_prefix_group, axis=1
            )
            all_prefix_outputs = [
                replace(group, tokens=embeddings)
                for group, embeddings in zip(prefix_groups, prefix_embeddings_split)
            ]
        else:
            all_prefix_outputs = []
        timestep_embeddings = einops.rearrange(
            timestep_embeddings,
            "batch (horizon n_tokens) d -> batch horizon n_tokens d",
            horizon=horizon,
        )

        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]
        timestep_embeddings_split = split_tokens(
            timestep_embeddings, tokens_per_timestep_group, axis=2
        )

        all_timestep_outputs = [
            replace(group, tokens=embeddings)
            for group, embeddings in zip(timestep_groups, timestep_embeddings_split)
        ]
        return all_prefix_outputs, all_timestep_outputs

    def assemble_input_tokens(
        self,
        prefix_groups: Sequence[PrefixGroup],
        timestep_groups: Sequence[TimestepGroup],
    ):
        """
        - Concatenate all timestep tokens together
        - Fold horizon dim into token sequence dim.
        - Prepend task tokens.
        """
        if len(prefix_groups) > 0:
            all_prefix_tokens = jnp.concatenate(
                [group.tokens for group in prefix_groups], axis=1
            )
        else:
            all_prefix_tokens = jnp.zeros(
                (
                    timestep_groups[0].tokens.shape[0],
                    0,
                    timestep_groups[0].tokens.shape[-1],
                ),
                dtype=jnp.float32,
            )

        all_timestep_tokens = jnp.concatenate(
            [group.tokens for group in timestep_groups], axis=2
        )
        all_timestep_tokens = einops.rearrange(
            all_timestep_tokens,
            "batch horizon n_tokens d -> batch (horizon n_tokens) d",
        )
        tokens = jnp.concatenate([all_prefix_tokens, all_timestep_tokens], axis=1)
        return tokens

    def generate_attention_mask(
        self,
        prefix_groups: Sequence[PrefixGroup],
        timestep_groups: Sequence[TimestepGroup],
        pad_mask: jnp.ndarray,
    ):
        """
        TODO: Need to update this docstring
        Args:
            tokens_per_group: A dictionary {group_name: num_tokens_for_group for group_name in computation_groups}
            horizon: Number of timesteps in the trajectory window.
            pad_mask: A boolean mask of shape (batch, horizon) indicating which timesteps are padding.
        Returns:
            attention_mask: A boolean mask of shape (batch, num_heads, total_tokens, total_tokens)

        Generate default attention mask for transformer call. The default attention mask
        is causal (tokens cannot attend to future tokens) and masks attention to past action
        tokens (since attending to previous actions can hurt performance).

        We generate an NxN mask where the nth row denotes which tokens the nth token
        can attend to.

        attention_mask[i, j] = 1 denotes that token j can attend to token i.
        attention_mask[i, j] = 0 denotes that token j cannot attend to token i.

        This function first creates a lower triangular matrix with past actions masked out.
        Then this causal mask is offset by a non-causal mask for the task tokens.

        For example, given the token sequence: [t_0, t_1, o_0, a_0, o_1, a_1, o_2, a_2]
        the attention mask would be:

        1 1 0 0 0 0 0 0
        1 1 0 0 0 0 0 0
        1 1 0 0 0 0 0 0
        1 1 1 0 0 0 0 0
        1 1 1 0 0 0 0 0
        1 1 1 0 1 0 0 0
        1 1 1 0 1 0 0 0
        1 1 1 0 1 0 1 0
        """

        def _get_position(i, tokens_per_elem):
            return np.searchsorted(np.cumsum(tokens_per_elem), i)

        horizon = pad_mask.shape[1]
        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]

        tokens_for_prefix = sum(tokens_per_prefix_group)
        tokens_per_time_step = sum(tokens_per_timestep_group)

        total_tokens = tokens_for_prefix + tokens_per_time_step * horizon
        attention_mask = np.zeros((total_tokens, total_tokens), dtype=int)

        def get_token_description(i):
            if i < tokens_for_prefix:
                position = _get_position(i, tokens_per_prefix_group)
                return TokenMetadata.create(prefix_groups[position], timestep=-1)

            i -= tokens_for_prefix
            timestep, i = divmod(i, tokens_per_time_step)
            position = _get_position(i, tokens_per_timestep_group)
            return TokenMetadata.create(timestep_groups[position], timestep)

        for i in range(total_tokens):  # Token attending
            for j in range(total_tokens):  # Token being attended to
                # description is a TokenMetadata(token_name, token_timestep, extra_info)
                description_i = get_token_description(i)
                description_j = get_token_description(j)
                mask = int(description_i.should_attend_to(description_j))
                attention_mask[i, j] = mask

        pad_attention_mask = self.generate_pad_attention_mask(
            pad_mask, tokens_per_time_step, tokens_for_prefix
        )
        attention_mask = jnp.logical_and(attention_mask, pad_attention_mask)
        return attention_mask

    def generate_pad_attention_mask(
        self, pad_mask, tokens_per_time_step, tokens_for_prefix
    ):
        """
        Generate attention mask that ignores padding. `pad_mask` has shape (batch, horizon) and
        records which time steps are padding. We first expand the mask to shape (batch, horizon * tokens_per_time_step)
        and then prepend a mask for the task prefix to get shape (batch, total_tokens).
        We broadcast to (batch, num_heads, total_tokens, total_tokens).
        """
        horizon = pad_mask.shape[1]
        total_tokens = tokens_for_prefix + tokens_per_time_step * horizon
        sequence_mask = jnp.repeat(pad_mask, tokens_per_time_step, axis=1)
        task_mask = jnp.ones((pad_mask.shape[0], tokens_for_prefix), dtype=int)
        full_mask = jnp.concatenate([task_mask, sequence_mask], axis=1)
        full_mask = jnp.broadcast_to(
            full_mask[:, None, None, :],
            (
                full_mask.shape[0],
                self.num_attention_heads,
                total_tokens,
                total_tokens,
            ),
        )
        return full_mask

    def pretty_print_attention_mask(
        self,
        prefix_groups: Sequence[PrefixGroup],
        timestep_groups: Sequence[TimestepGroup],
    ):
        logging.warning("Prefix groups:")
        for prefix_group in prefix_groups:
            logging.warning(
                "PrefixGroup(name=%s, shape=%s, attends_to=%s)",
                prefix_group.name,
                prefix_group.tokens.shape,
                prefix_group.attends_to,
            )
        logging.warning("Timestep groups:")
        for timestep_group in timestep_groups:
            logging.warning(
                "TimestepGroup(name=%s, shape=%s, attends_to=%s)",
                timestep_group.name,
                timestep_group.tokens.shape,
                timestep_group.attends_to,
            )

        import rich

        horizon = timestep_groups[0].tokens.shape[1]

        all_metadatas: Sequence[TokenMetadata] = []
        column_names = []

        for prefix_group in prefix_groups:
            column_names.append(
                f"{prefix_group.name} ({prefix_group.tokens.shape[1]} tokens)"
            )
            all_metadatas.append(TokenMetadata.create(prefix_group, timestep=-1))

        for ts in range(horizon):
            for timestep_group in timestep_groups:
                column_names.append(
                    f"t={ts} {timestep_group.name} ({timestep_group.tokens.shape[2]} tokens) "
                )
                all_metadatas.append(TokenMetadata.create(timestep_group, timestep=ts))

        rows = []
        for j in range(len(all_metadatas)):  # Token being attended to
            row = [column_names[j]]
            for i in range(len(all_metadatas)):  # Token attending
                description_i = all_metadatas[i]
                description_j = all_metadatas[j]
                mask = int(description_i.should_attend_to(description_j))
                row.append("x" if mask else " ")
            rows.append(row)

        table = rich.table.Table(
            "", *column_names, title="Attention Mask", show_header=True
        )
        for row in rows:
            table.add_row(*row)
        rich.print(table)
