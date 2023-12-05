# Written by Dibya
from typing import Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from orca.model.components.block_transformer import (
    AttentionRule,
    BlockTransformer,
    PrefixGroup,
    TimestepGroup,
)
from orca.utils.typing import Data, Sequence

posemb_init = nn.initializers.normal(stddev=0.02)


class OrcaTransformer(nn.Module):
    """
    This module forms the base of the ORCA model.

    The core idea is to run a causal transformer on the following sequence,

        [task, observation 0, observation 1, observation 2, ...]

    but with additional groups of tokens ("readouts") that provide
    a way of "reading out" the information in the transformer.

    For example, we may have an "action" readout that provides embeddings that are
    useful for predicting actions, and a "value" readout with embeddings that are useful for
    predicting values.


    The transformer is a blockwise-causal transformer, where each timestep only attends to the same or previous timesteps.

    When called, the module requests a set of readouts, and performs a forward pass of the transformer on the following sequence:

        [
        <task tokens>,
        <observation ts0 tokens>, <readout1 ts0 tokens>, <readout2 ts0 tokens>, ...
        <observation ts1 tokens>, <readout1 ts1 tokens>, <readout2 ts1 tokens>, ...
        ...
    ]

    The observation tokens attend to the task prefix, and to all observation tokens in the same or previous timesteps.
    Readouts attend to everything observation tokens do, but are not attended to by observation or task tokens. All
    tokens within the same group and same timestep (e.g. "observation ts0 tokens") fully attend to each other.

    By this design, each readout does not influence the computation happening in the task or observation tokens,
    and each readout is **independent of one another**. This allows us to hot-swap in different
    readouts at any time (e.g. we can run with the action readout or the value readout or both at the same time).


    Args:
        observations_tokenizers (Sequence[nn.Module]): List of flax modules for tokenizing the observations.
            The output of each tokenizer is concatenated to form the observation tokens.
        task_tokenizers (Sequence[nn.Module]): List of flax modules for tokenizing the task.
            The output of each tokenizer is concatenated to form the task token prefix.
        readouts (Dict[str, int]): Dictionary of {readout_name: n_tokens_for_readout}
        transformer_kwargs (Dict): Dictionary of kwargs to forward to BlockTransformer.
        token_embedding_size (int): Dimension of the token embeddings (default: 512)
        max_horizon (int): The maximum number of timesteps that the transformer can be run with.
    """

    observation_tokenizers: Dict[str, nn.Module]
    task_tokenizers: Dict[str, nn.Module]
    readouts: Dict[str, int]
    transformer_kwargs: Dict
    token_embedding_size: int = 512
    max_horizon: int = 1

    @nn.compact
    def __call__(
        self,
        observations: Data,
        tasks: Data,
        pad_mask: jax.Array,
        readouts: Optional[Sequence[str]] = None,
        train: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            observations: A dictionary containing observation data for a batch of trajectory windows.
                Each entry has shape (batch, horizon, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            pad_mask: A boolean mask of shape (batch, horizon) where False indicates a padded timestep.
            readouts: A list of readouts to compute. If None, defaults to all readouts. Must be a subset of the readouts specified in the model config.
            train: Whether model is being trained.
            verbose: If True, prints out the transformer structure.

        Returns:
            embedding_dict: A dictionary {
                    **{readout_name: embedding of shape (batch, horizon, n_tokens_for_readout, token_embedding_size) for k in readouts},
                    also includes the outputs corresponding to the task and observation tokens (although this probably isn't as useful)
                }

        Note: Horizon can be anything <= max_horizon.
        """
        if readouts is None:
            readouts = list(self.readouts.keys())

        # Check that all inputs are valid
        assert set(readouts).issubset(
            set(self.readouts.keys())
        ), "readouts must be a subset of those specified in the model config"

        batch_size, horizon = jax.tree_util.tree_leaves(observations)[0].shape[:2]
        assert horizon <= self.max_horizon, "horizon must be <= max_horizon"
        assert jax.tree_util.tree_all(
            jax.tree_map(lambda x: x.shape[1] == horizon, observations)
        ), "observations must have the same horizon"

        # Create inputs for the transformer
        all_prefix_groups = []
        all_timestep_groups = []

        all_task_names = [f"task_{name}" for name in self.task_tokenizers]
        all_obs_names = [f"obs_{name}" for name in self.observation_tokenizers]

        task_attention_rules = {
            task_name: AttentionRule.CAUSAL for task_name in all_task_names
        }  # Tasks attend to all other tasks

        observation_attention_rules = {
            name: AttentionRule.CAUSAL for name in all_task_names + all_obs_names
        }  # Observations attend to all tasks and previous observations causally

        # First, add the task tokens
        for name, tok in self.task_tokenizers.items():
            # Receive inputs from tokenizer and cast to embedding size
            task_tokens = tok(observations, tasks, train=train)
            task_tokens = nn.Dense(self.token_embedding_size)(task_tokens)

            # task_tokens shape is (batch, n_tokens, token_embedding_size)

            # Add positional embedding
            task_pos_embedding = self._create_positional_embedding(
                f"task_{name}", task_tokens.shape[1], prefix=True
            )
            task_tokens += task_pos_embedding

            all_prefix_groups.append(
                PrefixGroup(f"task_{name}", task_tokens, task_attention_rules)
            )

        # Next, add the observation tokens
        for name, tok in self.observation_tokenizers.items():
            # Receive inputs from tokenizer and cast to embedding size
            obs_tokens = tok(observations, tasks, train=train)
            obs_tokens = nn.Dense(self.token_embedding_size)(obs_tokens)
            # obs_tokens shape is (batch, horizon, n_tokens, token_embedding_size)

            # Add positional embedding
            obs_pos_embedding = self._create_positional_embedding(
                f"obs_{name}", obs_tokens.shape[2], prefix=False
            )
            obs_tokens += obs_pos_embedding[:, :horizon, :, :]

            all_timestep_groups.append(
                TimestepGroup(f"obs_{name}", obs_tokens, observation_attention_rules)
            )

        # Finally, add the readout tokens
        for readout_name in readouts:
            # Readouts do not correspond to any inputs, so we just create a bunch of zeros
            n_tokens_for_readout = self.readouts[readout_name]
            readout_tokens = jnp.zeros(
                (batch_size, horizon, n_tokens_for_readout, self.token_embedding_size)
            )

            # Add positional embedding
            readout_pos_embedding = self._create_positional_embedding(
                f"readout_{readout_name}", n_tokens_for_readout, prefix=False
            )
            readout_tokens += readout_pos_embedding[:, :horizon, :, :]

            attention_rules = {
                **{
                    name: AttentionRule.CAUSAL
                    for name in all_task_names + all_obs_names
                },
                f"readout_{readout_name}": AttentionRule.CAUSAL,
            }  # Attend to tasks, all previous observations, and your own previous readout tokens

            all_timestep_groups.append(
                TimestepGroup(
                    f"readout_{readout_name}",
                    readout_tokens,
                    attention_rules,
                )
            )

        # Run the transformer!
        assert (
            self.transformer_kwargs.get("add_position_embedding", False) is False
        ), "Already added positional embeddings to the tokens"

        prefix_outputs, timestep_outputs = BlockTransformer(**self.transformer_kwargs)(
            all_prefix_groups,
            all_timestep_groups,
            pad_mask,
            train=train,
            verbose=verbose,
        )

        outputs = dict()
        outputs.update({group.name: group.tokens for group in prefix_outputs})
        outputs.update(
            {
                group.name.removeprefix("readout_"): group.tokens
                for group in timestep_outputs
            }
        )

        if len(prefix_outputs) > 0:
            outputs["task"] = jnp.concatenate(
                [group.tokens for group in prefix_outputs], axis=-2
            )
        outputs["obs"] = jnp.concatenate(
            [
                group.tokens
                for group in timestep_outputs
                if group.name.startswith("obs_")
            ],
            axis=-2,
        )

        return outputs

    def _create_positional_embedding(self, name, n_tokens, prefix=False):
        if prefix:
            shape = (1, n_tokens, self.token_embedding_size)
        else:
            shape = (1, self.max_horizon, n_tokens, self.token_embedding_size)
        return self.param(
            f"{name}_pos_embedding",
            posemb_init,
            shape,
        )


class OrcaModel(nn.Module):
    """
    Bundles OrcaTransformer with various heads (useful for keeping all parameters in one place).
    """

    orca_transformer: OrcaTransformer
    heads: Dict[str, nn.Module]

    def __call__(self, observations, tasks, pad_mask, train=True, verbose=False):
        """Run transformer and the main method for all heads. Useful for init.

        Args:
            observations: A dictionary containing observation data
                where each element has shape (batch, horizon, *).
            tasks: A dictionary containing task data
                where each element has shape (batch, *).
            pad_mask: A boolean mask of shape (batch, horizon) where False indicates a padded timestep.
            train: Run in training mode
            verbose: If True, prints out the structure of the OrcaTransformer

        Returns:
            transformer_embeddings: See OrcaTransformer.__call__
            head_outputs: dictionary of outputs from heads {head_name: output}
        """
        transformer_embeddings = self.orca_transformer(
            observations, tasks, pad_mask, train=train, verbose=verbose
        )
        head_outputs = {}
        for head_name, head in self.heads.items():
            head_outputs[head_name] = head(transformer_embeddings, train=train)
        return transformer_embeddings, head_outputs

    def run_transformer(self, *args, **kwargs):
        """Run transformer and return embeddings. See OrcaTransformer.__call__"""
        return self.orca_transformer(*args, **kwargs)

    def run_head(
        self,
        *args,
        head_name: str,
        head_method_name: str = "__call__",
        **kwargs,
    ):
        """A convenience utility to run a method on a single head.

        Args:
            head_name: Name of head to run.
            head_method_name: Name of method to run on head. Defaults to "__call__".
            train: Whether model is being trained.
            **kwargs: Keyword arguments to pass to method.
        """
        head = self.heads[head_name]
        method = getattr(head, head_method_name)
        return method(*args, **kwargs)
