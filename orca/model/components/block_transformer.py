# Written by Dibya
from dataclasses import asdict, dataclass, replace
from enum import Enum
import logging
from typing import Mapping, Tuple

import einops
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.components.transformer import Transformer
from orca.utils.typing import Sequence, Union


class AttentionRule(Enum):
    """Enum describing when to attend to another token group.
    For most use cases, you should use WhenToAttend.CAUSAL or WhenToAttend.NEVER.
    """

    NEVER = "never"
    CAUSAL = "other.timestep <= self.timestep"
    CURRENT = "other.timestep == self.timestep"
    STRICT_PAST = "other.timestep < self.timestep"
    ALL = "all"  # Breaks causal structure! Be careful


@dataclass
class PrefixGroup:
    """A group of tokens that will be at the beginning of the token sequence. (e.g. task tokens)"""

    name: str
    tokens: jax.typing.ArrayLike  # with shape (batch, n_tokens, token_embedding_size)
    attention_rules: Mapping[str, AttentionRule]

    def __post_init__(self):
        assert self.tokens.ndim == 3, "PrefixGroup tokens must be (batch, n_tokens, d)"


@dataclass
class TimestepGroup:
    """A group of tokens that is repeated for each timestep. (e.g. observation tokens)"""

    name: str
    tokens: jax.typing.ArrayLike  # with shape (batch, horizon, n_tokens, token_embedding_size)
    attention_rules: Mapping[str, AttentionRule]

    def __post_init__(self):
        assert (
            self.tokens.ndim == 4
        ), "TimestepGroup tokens must be (batch, horizon, n_tokens, d))"


@dataclass
class TokenMetadata:
    """Useful metadata for computing attention masks. Note that all tokens within the
    same group at the same timestep always attend to each other unless you explicitly have
    attention_rules[self.name] = AttentionRule.NEVER
    """

    name: str
    timestep: int  # -1 for prefix tokens
    attention_rules: Mapping[str, AttentionRule]

    @classmethod
    def create(cls, group: Union[PrefixGroup, TimestepGroup], timestep: int):
        group_dict = asdict(group)
        group_dict.pop("tokens")
        return cls(
            timestep=timestep,
            **group_dict,
        )

    def should_attend_to(self, other_metadata: "TokenMetadata") -> bool:
        attention_rule = self.attention_rules.get(
            other_metadata.name, AttentionRule.NEVER
        )

        if attention_rule == AttentionRule.CAUSAL:
            return other_metadata.timestep <= self.timestep
        elif attention_rule == AttentionRule.CURRENT:
            return other_metadata.timestep == self.timestep
        elif attention_rule == AttentionRule.STRICT_PAST:
            return other_metadata.timestep < self.timestep
        elif attention_rule == AttentionRule.ALL:
            return True
        elif attention_rule == AttentionRule.NEVER:
            return False
        else:
            raise ValueError(f"Invalid attention rule: {attention_rule}")


def split_tokens(ary, n_tokens_per_group, axis):
    cumsum = np.cumsum(n_tokens_per_group)
    return jnp.split(ary, cumsum, axis=axis)


class BlockTransformer(nn.Module):
    # Forwarded to Transformer
    num_layers: int = 4
    mlp_dim: int = 1024
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = False

    # Enforce that timestep causal structure is not broken (future timesteps can't attend to past timesteps)
    enforce_causal: bool = True

    @nn.compact
    def __call__(
        self,
        prefix_groups: Sequence[PrefixGroup],
        timestep_groups: Sequence[TimestepGroup],
        timestep_pad_mask: jax.typing.ArrayLike,
        train: bool,
        verbose: bool = False,
    ) -> Tuple[Sequence[PrefixGroup], Sequence[TimestepGroup]]:
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
            prefix_outputs: A list of PrefixGroup objects containing the output embeddings for each token group.
            timestep_outputs: A list of TimestepGroup objects containing the output embeddings for each token group.
        """
        if verbose:
            self.pretty_print_attention_mask(prefix_groups, timestep_groups)

        horizon = timestep_groups[0].tokens.shape[1]
        assert all([group.tokens.shape[1] == horizon for group in timestep_groups])

        token_dim = timestep_groups[0].tokens.shape[-1]
        assert all([group.tokens.shape[-1] == token_dim for group in prefix_groups])
        assert all([group.tokens.shape[-1] == token_dim for group in timestep_groups])

        # Creates correct attention mask for transformer using group attention rules
        attention_mask = self.generate_attention_mask(
            prefix_groups, timestep_groups, timestep_pad_mask
        )

        # Assemble input tokens (batch, total_tokens, token_embedding_size)
        input_tokens = self.assemble_input_tokens(prefix_groups, timestep_groups)

        # Run transformer
        transformer = Transformer(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_attention_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            add_position_embedding=self.add_position_embedding,
        )
        output = transformer(input_tokens, attention_mask, train=train)

        # Split output into prefix and timestep groups

        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        n_prefix_tokens = sum(tokens_per_prefix_group)

        prefix_embeddings, timestep_embeddings = jnp.split(
            output, [n_prefix_tokens], axis=1
        )

        # Process prefix group outputs
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

        # Process timestep group outputs
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

        Returns:
            tokens: A tensor of shape (batch, total_tokens, token_embedding_size)
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
        timestep_pad_mask: jax.typing.ArrayLike,
    ):
        """
        Args:
            prefix_groups: A list of PrefixGroup objects.
            timestep_groups: A list of TimestepGroup objects.
            pad_mask: A boolean mask of shape (batch, horizon) indicating which timesteps are padding.

        Returns:
            attention_mask: A boolean mask of shape (batch, num_heads, total_tokens, total_tokens)

        We use the attention rules within each group to determine the transformer attention mask.
        """

        if self.enforce_causal:
            # First verify that prefix group isn't attending to any timestep group
            for prefix_group in prefix_groups:
                for ts_group in timestep_groups:
                    assert (
                        prefix_group.attention_rules.get(
                            ts_group.name, AttentionRule.NEVER
                        )
                        == AttentionRule.NEVER
                    ), f"Causality broken! Prefix group {prefix_group.name} is attending to timestep group {ts_group.name}"

            # Next, make sure that timestep groups aren't attending to future timesteps
            for group in prefix_groups + timestep_groups:
                for rule in group.attention_rules.values():
                    assert (
                        rule != AttentionRule.ALL
                    ), "Causality broken! WhenToAttend.ALL attends to future timesteps too."

        def _get_position(i, tokens_per_elem):
            return np.searchsorted(np.cumsum(tokens_per_elem), i)

        horizon = timestep_pad_mask.shape[1]
        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]

        tokens_for_prefix = sum(tokens_per_prefix_group)
        tokens_per_time_step = sum(tokens_per_timestep_group)

        total_tokens = tokens_for_prefix + tokens_per_time_step * horizon
        attention_mask = np.zeros((total_tokens, total_tokens), dtype=int)

        def get_token_metadata(i):
            if i < tokens_for_prefix:
                position = _get_position(i, tokens_per_prefix_group)
                return TokenMetadata.create(prefix_groups[position], timestep=-1)

            i -= tokens_for_prefix
            timestep, i = divmod(i, tokens_per_time_step)
            position = _get_position(i, tokens_per_timestep_group)
            return TokenMetadata.create(timestep_groups[position], timestep)

        for i in range(total_tokens):  # Token attending
            for j in range(total_tokens):  # Token being attended to
                metadata_i = get_token_metadata(i)
                metadata_j = get_token_metadata(j)
                mask = int(metadata_i.should_attend_to(metadata_j))
                attention_mask[i, j] = mask

        pad_attention_mask = self.generate_pad_attention_mask(
            timestep_pad_mask, tokens_per_time_step, tokens_for_prefix
        )
        attention_mask = jnp.logical_and(attention_mask, pad_attention_mask)
        return attention_mask

    def generate_pad_attention_mask(
        self,
        timestep_pad_mask: jax.typing.ArrayLike,
        tokens_per_time_step: int,
        tokens_for_prefix: int,
    ):
        """
        Generate attention mask that ignores padding. `timestep_pad_mask` has shape (batch, horizon) and
        records which time steps are padding. We first expand the mask to shape (batch, horizon * tokens_per_time_step)
        and then prepend a mask for the task prefix to get shape (batch, total_tokens).
        We broadcast to (batch, num_heads, total_tokens, total_tokens).
        """
        batch_size, horizon = timestep_pad_mask.shape

        total_tokens = tokens_for_prefix + tokens_per_time_step * horizon
        sequence_mask = jnp.repeat(timestep_pad_mask, tokens_per_time_step, axis=1)
        task_mask = jnp.ones((batch_size, tokens_for_prefix), dtype=int)
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
                flax.core.frozen_dict.pretty_repr(prefix_group.attention_rules),
            )
        logging.warning("Timestep groups:")
        for timestep_group in timestep_groups:
            logging.warning(
                "TimestepGroup(name=%s, shape=%s, attends_to=%s)",
                timestep_group.name,
                timestep_group.tokens.shape,
                flax.core.frozen_dict.pretty_repr(timestep_group.attention_rules),
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
                metadata_i = all_metadatas[i]
                metadata_j = all_metadatas[j]
                mask = int(metadata_i.should_attend_to(metadata_j))
                row.append("x" if mask else " ")
            rows.append(row)

        table = rich.table.Table(
            rich.table.Column(no_wrap=True),
            *column_names,
            title="Attention Mask",
            show_header=True,
            show_lines=True,
        )
        for row in rows:
            table.add_row(*row)
        rich.print(table)
