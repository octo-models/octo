# Written by Dibya
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.components import TokenMetadata, TokenType
from orca.model.components.computation_groups import ComputationPlaceholder
from orca.model.components.transformer import Transformer
from orca.utils.typing import Dict, PRNGKey, Sequence

# with shape (batch, horizon, n_tokens, token_embedding_size)
TransformerInputs = jnp.ndarray


class ORCATransformer(nn.Module):
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
        <observation ts0 tokens>, <computation_group1 ts0 tokens>, <computation_group2 ts0 tokens>, ...
        <observation ts1 tokens>, <computation_group1 ts1 tokens>, <computation_group2 ts1 tokens>, ...
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
    token_embedding_size: int = 512
    max_horizon: int = 1
    num_layers: int = 4
    mlp_dim: int = 1024
    num_heads: int = 8
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    def setup(self):
        self.transformer = Transformer(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            add_position_embedding=False,  # we add our own
        )

        self.tokens_per_obs = sum(tok.num_tokens for tok in self.observation_tokenizers)
        self.tokens_per_task = sum(tok.num_tokens for tok in self.task_tokenizers)

        self.obs_proj = [
            nn.Dense(self.token_embedding_size)
            for _ in range(len(self.observation_tokenizers))
        ]

        posemb_init = nn.initializers.normal(stddev=0.02)  # from BERT.
        self.obs_embedding = self.param(
            "obs_pos_embedding",
            posemb_init,
            (1, self.max_horizon, self.tokens_per_obs, self.token_embedding_size),
        )

        self.task_proj = [
            nn.Dense(self.token_embedding_size)
            for _ in range(len(self.task_tokenizers))
        ]

        self.task_embedding = self.param(
            "task_pos_embedding",
            posemb_init,
            (1, self.tokens_per_task, self.token_embedding_size),
        )

    def __call__(
        self,
        observations,
        tasks,
        computation_groups: Dict[str, TransformerInputs],
        train: bool = False,
    ):
        """
        Performs a forward pass of the network with certain computation groups and returns the corresponding embeddings

        Note: By construction, computation groups are independent of one another! The following two calls are equivalent:
        ```
        transformer_embeddings1 = model(observations, tasks,
            computation_groups={"actions": action_tokens})
        transformer_embeddings2 = model(observations, tasks,
            computation_groups={"actions": action_tokens, "value": value_tokens})
        transformer_embeddings1["action"] == transformer_embeddings2["action"]
        ```

        Args:
            observations: A dictionary containing observation data for a batch of trajectory windows.
                Each entry has shape (batch, horizon, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            computation_groups: A dictionary {string: transformer_inputs} where transformer_inputs
                has shape (batch, horizon, n_tokens, token_embedding_size)
                (n_tokens may vary between different computation groups)
            train: Whether to use dropout.

        Returns:
            embedding_dict: A dictionary {
                    "task": task_embeddings, # shape (batch, tokens_per_task, token_embedding_size)
                    "obs": obs_embeddings, # shape (batch, horizon, tokens_per_obs, token_embedding_size)
                    **{k: embedding for k in computation_groups} # shape (batch, horizon, computation_groups[k].shape[-2], token_embedding_size)
                }

        Note: Horizon can be anything <= max_horizon.
        """

        horizon = next(iter(jax.tree_util.tree_leaves(observations))).shape[1]
        assert horizon <= self.max_horizon, "horizon must be <= max_horizon"
        assert jax.tree_util.tree_all(
            jax.tree_map(lambda x: x.shape[1] == horizon, observations)
        ), "observations must have the same horizon"
        assert jax.tree_util.tree_all(
            jax.tree_map(lambda x: x.shape[1] == horizon, computation_groups)
        ), "computation_groups must have the same horizon"

        # tokens_per_group is {group_name: num_tokens_for_group for group_name in computation_groups}
        tokens_per_group = {k: v.shape[2] for k, v in computation_groups.items()}
        tokens_per_time_step = self.tokens_per_obs + sum(tokens_per_group.values())

        attention_mask = self.generate_attention_mask(
            tokens_per_group, horizon, observations["pad_mask"]
        )

        # task_tokens has shape (batch, tokens_per_task, token_embedding_size)
        # obs_tokens has shape (batch, horizon, tokens_per_obs, token_embedding_size)
        task_tokens, obs_tokens = self.tokenize_observations_and_tasks(
            observations, tasks, train=train
        )

        # input_tokens has shape (batch, tokens_per_task + horizon * tokens_per_time_step, token_embedding_size)
        input_tokens = self.assemble_input_tokens(
            task_tokens, obs_tokens, computation_groups
        )

        # Run the transformer! output has same shape as input_tokens
        output = self.transformer(input_tokens, attention_mask, train=train)

        all_embeddings = {}

        # first, we extract the embeddings corresponding to the task prefix
        all_embeddings["task"] = output[:, : self.tokens_per_task]
        output = output[:, self.tokens_per_task :]

        # The remaining embeddings are repeated per timestep, so we unfold horizon length from token sequence length
        output = jnp.reshape(
            output,
            (
                output.shape[0],
                horizon,
                tokens_per_time_step,
                self.token_embedding_size,
            ),
        )

        # The first tokens_per_obs tokens are the observation tokens
        all_embeddings["obs"] = output[:, :, : self.tokens_per_obs]
        output = output[:, :, self.tokens_per_obs :]

        # The remaining tokens are the computation group tokens

        token_boundaries = np.cumsum(list(tokens_per_group.values()))
        # get the embeddings for each computation group
        output_per_group = jnp.split(output, token_boundaries, axis=2)

        # Assign each group its corresponding name
        for i, group_name in enumerate(tokens_per_group):
            all_embeddings[group_name] = output_per_group[i]
        return all_embeddings

    def tokenize_observations_and_tasks(self, observations, tasks, train: bool = False):
        """
        Tokenize observation/action history and task (either goal image or language).

        Args:
            observations: A dictionary containing observation data for a batch of trajectory windows.
                Each entry has shape (batch, horizon, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            train: Whether to use dropout.
        """

        # a list of (batch, horizon, tokens_per_obs_tokenizer, token_embedding_size)
        obs_tokens = [
            proj(tok(observations, tasks, train=train))
            for tok, proj in zip(self.observation_tokenizers, self.obs_proj)
        ]
        # (batch, horizon, tokens_per_obs, token_embedding_size)
        obs_tokens = jnp.concatenate(obs_tokens, axis=-2)
        # Add positional embedding to obs tokens
        obs_tokens += self.obs_embedding[:, : obs_tokens.shape[1]]

        if len(self.task_tokenizers) > 0:
            # a list of (batch, tokens_per_task_tokenizer, token_embedding_size)
            task_tokens = [
                proj(tok(observations, tasks, train=train))
                for tok, proj in zip(self.task_tokenizers, self.task_proj)
            ]
            # (batch, tokens_per_task, token_embedding_size)
            task_tokens = jnp.concatenate(task_tokens, axis=-2)
        else:
            task_tokens = jnp.zeros(
                (obs_tokens.shape[0], self.tokens_per_task, self.token_embedding_size)
            )
        # Add positional embedding to task tokens
        task_tokens += self.task_embedding[:, : task_tokens.shape[1]]

        assert obs_tokens.shape[-2] == self.tokens_per_obs
        assert task_tokens.shape[-2] == self.tokens_per_task

        return task_tokens, obs_tokens

    def assemble_input_tokens(
        self,
        task_tokens,
        obs_tokens,
        computation_groups: Dict[str, TransformerInputs],
    ):
        """
        Concatenate obs and computation group tokens.
        Fold horizon dim into token sequence dim.
        Prepend task tokens.
        """

        # (batch, horizon, tokens_per_time_step, token_embedding_size)
        tokens = jnp.concatenate(
            [obs_tokens] + [computation_groups[k] for k in computation_groups], axis=2
        )
        # (batch, horizon * tokens_per_time_step, token_embedding_size)
        tokens = jnp.reshape(tokens, (tokens.shape[0], -1, tokens.shape[-1]))

        # (batch, tokens_per_task + horizon * tokens_per_time_step, token_embedding_size)
        tokens = jnp.concatenate([task_tokens, tokens], axis=1)
        return tokens

    def generate_attention_mask(self, tokens_per_group, horizon, pad_mask):
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

        tokens_per_time_step = self.tokens_per_obs + sum(tokens_per_group.values())

        total_tokens = self.tokens_per_task + tokens_per_time_step * horizon
        attention_mask = np.zeros((total_tokens, total_tokens), dtype=int)

        def get_token_description(i):
            if i < self.tokens_per_task:
                return TokenMetadata(TokenType.TASK, 0, 0)
            else:
                i -= self.tokens_per_task
                timestep, position = divmod(i, tokens_per_time_step)
                if position < self.tokens_per_obs:
                    return TokenMetadata(TokenType.OBS, timestep)
                else:
                    group_number = np.searchsorted(
                        np.cumsum(list(tokens_per_group.values())),
                        position - self.tokens_per_obs,
                    )
                    return TokenMetadata(
                        TokenType.MISC, timestep, dict(group_number=group_number)
                    )

        for i in range(total_tokens):  # Token attending
            for j in range(total_tokens):  # Token being attended to
                # description is a TokenMetadata(token_name, token_timestep, extra_info)
                description_i = get_token_description(i)
                description_j = get_token_description(j)

                if description_i.kind == TokenType.TASK:
                    # Only attend to other task tokens
                    mask = 1 if description_j.kind == TokenType.TASK else 0
                elif description_i.kind == TokenType.OBS:
                    # Only attend to observation tokens in the same timestep or before
                    if description_j.kind == TokenType.TASK:
                        mask = 1
                    elif description_j.kind == TokenType.OBS:
                        mask = (
                            1 if description_j.timestep <= description_i.timestep else 0
                        )
                    else:
                        mask = 0  # Don't attend to actions
                else:
                    if description_j.kind == TokenType.TASK:
                        mask = 1
                    elif description_j.kind == TokenType.OBS:
                        mask = (
                            1 if description_j.timestep <= description_i.timestep else 0
                        )
                    else:
                        # Only attend to tokens in the same group
                        mask = (
                            1
                            if description_j.extra_metadata["group_number"]
                            == description_i.extra_metadata["group_number"]
                            else 0
                        )

                attention_mask[i, j] = mask

        pad_attention_mask = self.generate_pad_attention_mask(
            pad_mask, tokens_per_time_step
        )
        attention_mask = jnp.logical_and(attention_mask, pad_attention_mask)
        return attention_mask

    def generate_pad_attention_mask(self, pad_mask, tokens_per_time_step):
        """
        Generate attention mask that ignores padding. `pad_mask` has shape (batch, horizon) and
        records which time steps are padding. We first expand the mask to shape (batch, horizon * tokens_per_time_step)
        and then prepend a mask for the task prefix to get shape (batch, total_tokens).
        We broadcast to (batch, num_heads, total_tokens, total_tokens).
        """
        horizon = pad_mask.shape[1]
        total_tokens = self.tokens_per_task + tokens_per_time_step * horizon
        sequence_mask = jnp.repeat(pad_mask, tokens_per_time_step, axis=1)
        task_mask = jnp.ones((pad_mask.shape[0], self.tokens_per_task), dtype=int)
        full_mask = jnp.concatenate([task_mask, sequence_mask], axis=1)
        full_mask = jnp.broadcast_to(
            full_mask[:, None, None, :],
            (
                full_mask.shape[0],
                self.num_heads,
                total_tokens,
                total_tokens,
            ),
        )
        return full_mask


class OrcaModel(nn.Module):
    """
    Wrapper class for ORCATransformer that bundles computation placeholders
    and heads with the base transformer (useful for keeping all parameters in one place).
    """

    computation_placeholders: Dict[str, ComputationPlaceholder]
    orca_transformer: ORCATransformer
    heads: Dict[str, nn.Module]

    def __call__(self, observations, tasks, computation_groups=None, *, train):
        """Runs the base transformer (using default computation groups if not provided).

        Args:
            observations: A dictionary containing observation data
                where each element has shape (batch, horizon, *).
            tasks: A dictionary containing task data
                where each element has shape (batch, *).
            computation_groups: If not provided, uses the default computation placeholders
                bundled in OrcaModel.
            train: Whether model is being trained.
        Returns:
            transformer_embeddings: See ORCATransformer.__call__. Has elements: {
                "task": task_embeddings, # shape (batch, tokens_per_task, token_embedding_size)
                "obs": obs_embeddings, # shape (batch, horizon, tokens_per_obs, token_embedding_size)
                **{k: embedding for k in computation_groups} # shape (batch, horizon, computation_groups[k].shape[-2], token_embedding_size)
            }
        """
        if computation_groups is None:
            computation_groups = self.get_default_computation_groups(
                observations, tasks
            )
        return self.orca_transformer(
            observations, tasks, computation_groups, train=train
        )

    def get_default_computation_groups(self, observations, tasks):
        batch_size, horizon = jax.tree_util.tree_leaves(observations)[0].shape[:2]
        computation_groups = {
            k: placeholder(batch_size, horizon)
            for k, placeholder in self.computation_placeholders.items()
        }
        return computation_groups

    def run_head(
        self,
        observations,
        tasks,
        *args,
        head_name: str,
        computation_group_name: str,
        head_method_name: str = "__call__",
        computation_groups=None,
        train=True,
        **kwargs,
    ):
        """A convenience utility to run the transformer and a single head after.

        Not recommended if you want to run multiple heads on the transformer or run the transformer without any heads.
        (See train.py for an example of how to do this.)

        Args:
            observations: A dictionary containing observation data
                where each element has shape (batch, horizon, *).
            tasks: A dictionary containing task data
                where each element has shape (batch, *).
            computation_groups: See __call__.
            train: Whether model is being trained.

            head_name: Name of head to run.
            computation_group_name: Which transformer embedding to pass to head.
            head_method_name: Name of method to run on head. Defaults to "__call__".
            *args: Additional arguments to pass to method.
            **kwargs: Keyword arguments to pass to method.
        """
        if computation_groups is None:
            computation_groups = self.get_default_computation_groups(
                observations, tasks
            )
            # TODO: this can be made more efficient: you only actually ever
            # have to pass in the computation group name that the head needs.

        # Run the transformer!
        transformer_embeddings = self(
            observations, tasks, computation_groups, train=train
        )

        # Extract relevant embeddings for the head
        embeddings = transformer_embeddings[computation_group_name]

        # Run the head!
        head = self.heads[head_name]
        head_method_name = head_method_name or "__call__"
        return getattr(head, head_method_name)(embeddings, *args, train=train, **kwargs)
