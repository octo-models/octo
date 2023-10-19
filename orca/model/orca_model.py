# Written by Dibya
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.components.block_transformer import (
    BlockTransformer,
    PrefixGroup,
    TimestepGroup,
)
from orca.utils.typing import Dict, PRNGKey, Sequence

# with shape (batch, horizon, n_tokens, token_embedding_size)
TransformerInputs = jnp.ndarray

posemb_init = nn.initializers.normal(stddev=0.02)


class OrcaTransformer(nn.Module):
    """
    This module forms the base of the ORCA model.

    The core idea is to run a causal transformer on the following sequence,

        [task, observation 0, observation 1, observation 2, ...]

    but with additional groups of tokens (computational groups) that provide
    a way of "reading out" the information in the transformer.

    For example, we may have a "action" computational group that reads out
    transformer embeddings that are useful for predicting actions, and a "value"
    computational group that reads out transformer embeddings that are useful for
    predicting values.


    The transformer is a blockwise-causal transformer, where each timestep only attends to the same or previous timesteps.

    When called, the module requests a set of computation groups, and performs a forward pass of the transformer on the following sequence:

        [
        task,
        <observation ts0 tokens>, < ts0 tokens>, <readout_group2 ts0 tokens>, ...
        <observation ts1 tokens>, <readout_group1 ts1 tokens>, <readout_group2 ts1 tokens>, ...
        ...
    ]

    The observation tokens attend to the task prefix, and to all observation tokens in the same or previous timesteps.
    Computation group tokens attend to everything observation tokens do, as well as computation group tokens in the same group and same timestep.

    By this design, each computational group does not influence the computation
    happening in the task or observation tokens, and each group is **independent* of one another**.
    This allows us to hot-swap in different computational groups at any time (e.g. we can run
    with the action computational group or the value computational group or both at the same time).


    Args:
        observations_tokenizers (Sequence[nn.Module]): List of flax modules for tokenizing the observations.
            The output of each tokenizer is concatenated to form the observation tokens.
        task_tokenizers (Sequence[nn.Module]): List of flax modules for tokenizing the task.
            The output of each tokenizer is concatenated to form the task token prefix.
        token_embedding_size (int): Dimension of the token embeddings (default: 512)
        max_horizon (int): Number of timesteps in the trajectory window.
        num_layers (int): Number of layers in transformer.
        mlp_dim (int): MLP dim of transformer.
        num_heads (int): Number of attention heads in transformer.
        dropout_rate (float): Dropout rate for transformer
        attention_dropout_rate (float): Dropout in self-attention."""

    observation_tokenizers: Sequence[nn.Module]
    task_tokenizers: Sequence[nn.Module]
    readouts: Dict[str, int]
    transformer_kwargs: Dict
    token_embedding_size: int = 512
    max_horizon: int = 1

    @nn.compact
    def __call__(
        self,
        observations,
        tasks,
        pad_mask,
        readouts: Sequence[str] = None,
        train: bool = False,
    ):
        """
        Performs a forward pass of the network with certain computation groups and returns the corresponding embeddings

        Note: By construction, computation groups are independent of one another! The following two calls are equivalent:
        ```
        transformer_embeddings1 = model(observations, tasks,
            readout_groups={"actions": action_tokens})
        transformer_embeddings2 = model(observations, tasks,
            readout_groups={"actions": action_tokens, "value": value_tokens})
        transformer_embeddings1["action"] == transformer_embeddings2["action"]
        ```

        Args:
            observations: A dictionary containing observation data for a batch of trajectory windows.
                Each entry has shape (batch, horizon, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            readout_groups: A dictionary {string: transformer_inputs} where transformer_inputs
                has shape (batch, horizon, n_tokens, token_embedding_size)
                (n_tokens may vary between different computation groups)
            train: Whether to use dropout.

        Returns:
            embedding_dict: A dictionary {
                    "task": task_embeddings, # shape (batch, tokens_per_task, token_embedding_size)
                    "obs": obs_embeddings, # shape (batch, horizon, tokens_per_obs, token_embedding_size)
                    **{k: embedding for k in readout_groups} # shape (batch, horizon, readout_groups[k].shape[-2], token_embedding_size)
                }

        Note: Horizon can be anything <= max_horizon.
        """
        if readouts is None:
            readouts = list(self.readouts.keys())

        assert set(readouts).issubset(
            set(self.readouts.keys())
        ), "readout_groups must be a subset of the readouts specified in the model config"

        batch_size, horizon = jax.tree_util.tree_leaves(observations)[0].shape[:2]
        assert horizon <= self.max_horizon, "horizon must be <= max_horizon"
        assert jax.tree_util.tree_all(
            jax.tree_map(lambda x: x.shape[1] == horizon, observations)
        ), "observations must have the same horizon"

        all_task_names = [f"task{k}" for k in range(len(self.task_tokenizers))]
        all_obs_names = [f"obs{k}" for k in range(len(self.observation_tokenizers))]

        all_prefix_groups = []

        # First, parse the tasks
        for k, tok in enumerate(self.task_tokenizers):
            task_tokens = tok(observations, tasks, train=train)
            task_tokens = nn.Dense(self.token_embedding_size)(task_tokens)
            task_pos_embedding = self.param(
                f"task{k}_pos_embedding",
                posemb_init,
                (1, task_tokens.shape[1], self.token_embedding_size),
            )
            task_tokens += task_pos_embedding
            all_prefix_groups.append(
                PrefixGroup(f"task{k}", task_tokens, attends_to=all_task_names)
            )

        all_timestep_groups = []
        for k, tok in enumerate(self.observation_tokenizers):
            obs_tokens = tok(observations, tasks, train=train)
            obs_tokens = nn.Dense(self.token_embedding_size)(obs_tokens)
            obs_pos_embedding = self.param(
                f"obs{k}_pos_embedding",
                posemb_init,
                (1, self.max_horizon, obs_tokens.shape[2], self.token_embedding_size),
            )
            obs_tokens += obs_pos_embedding[:, :horizon, :, :]

            all_timestep_groups.append(
                TimestepGroup(
                    f"obs{k}", obs_tokens, attends_to=all_task_names + all_obs_names
                )
            )

        for readout_name in readouts:
            n_tokens_for_readout = self.readouts[readout_name]
            readout_embeddings = self.param(
                f"readout_{readout_name}_embedding",
                posemb_init,
                (1, self.max_horizon, n_tokens_for_readout, self.token_embedding_size),
            )
            readout_embeddings = jnp.broadcast_to(
                readout_embeddings[:, :horizon, :, :],
                (batch_size, horizon, n_tokens_for_readout, self.token_embedding_size),
            )
            all_timestep_groups.append(
                TimestepGroup(
                    readout_name,
                    readout_embeddings,
                    attends_to=all_task_names + all_obs_names,
                )
            )

        prefix_outputs, timestep_outputs = BlockTransformer(**self.transformer_kwargs)(
            all_prefix_groups, all_timestep_groups, pad_mask, train=train
        )

        return {
            **{group.name: group.tokens for group in prefix_outputs},
            **{group.name: group.tokens for group in timestep_outputs},
        }


class OrcaModel(nn.Module):
    """
    Wrapper class for ORCATransformer that bundles computation placeholders
    and heads with the base transformer (useful for keeping all parameters in one place).
    """

    orca_transformer: OrcaTransformer
    heads: Dict[str, nn.Module]

    def __call__(self, *args, **kwargs):
        return self.orca_transformer(*args, **kwargs)

    def run_head(
        self,
        observations,
        tasks,
        pad_mask,
        *head_method_args,
        head_name: str,
        readout_name: str,
        head_method_name: str = "__call__",
        train=True,
        **head_method_kwargs,
    ):
        """A convenience utility to run the transformer and a single head after.

        Not recommended if you want to run multiple heads on the transformer or run the transformer without any heads.
        (See train.py for an example of how to do this.)

        Args:
            observations: A dictionary containing observation data
                where each element has shape (batch, horizon, *).
            tasks: A dictionary containing task data
                where each element has shape (batch, *).
            readout_groups: See __call__.
            train: Whether model is being trained.

            head_name: Name of head to run.
            readout_group_name: Which transformer embedding to pass to head.
            head_method_name: Name of method to run on head. Defaults to "__call__".
            *args: Additional arguments to pass to method.
            **kwargs: Keyword arguments to pass to method.
        """

        transformer_embeddings = self.orca_transformer(
            observations, tasks, pad_mask, train=train
        )

        # Extract relevant embeddings for the head
        embeddings = transformer_embeddings[readout_name]

        # Run the head!
        head = self.heads[head_name]
        method = getattr(head, head_method_name)
        return method(embeddings, *head_method_args, train=train, **head_method_kwargs)
