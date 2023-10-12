# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.components.action_heads import DiscretizedActionHead
from orca.model.components.tokenizers import ActionTokenizer
from orca.model.components.transformer import Transformer
from orca.utils.typing import PRNGKey, Sequence


class ORCAPolicy(nn.Module):
    """
    This transformer models a sequence of observations and actions (of length `horizon`),
    prefixed by a task (either a goal image or language instruction). The observations,
    actions, and task are tokenized and then assembled into a token sequence.

    With no action chunking (`pred_horizon=1`) the token sequence looks like:

    [
        task,
        o_0, <placeholder a_0>,
        o_1, <placeholder a_1>,
        ...,
        o_n, <placeholder a_n>
     ]

    With action chunking (`pred_horizon>1`), the actions in the chunk are packed between
    the observations. For example, if `pred_horizon=3`, the sequence looks like:

    [
        task,
        o_0, <placeholder a_0>, <placeholder a_1>, <placeholder a_2>,
        o_1, <placeholder a_1>, <placeholder a_2>, <placeholder a_3>,
        ...,
        o_n, <placeholder a_n>, <placeholder a_n+1>, <placeholder a_n+2>
    ]

        (See `DiscretizedActionHead` for more details on the action placeholders.)

    In both cases, we use a causal mask to ensures that action token
    prediction can only attend to *past* observation and task tokens.

    At test time, we predict actions non-autoregressively (i.e we predict all the tokens
    of all the actions in a chunk in one forward pass). We do this by using mask/padding tokens
    in the input sequence in place of the action tokens. Thus, all the action tokens are masked
    at training time as well. Note: even though the attention mask ensures we don't attend to the
    actions, we still need to mask them because of the residual connection in the attention block.
    When rolling out the policy, we left pad the observation-action history until we have enough
    history to fill the entire context window.

    Args:
        observations_tokenizers (Sequence[nn.Module]): List of flax modules for tokenizing the observations.
            The output of each tokenizer is concatenated to form the observation tokens.
        task_tokenizers (Sequence[nn.Module]): List of flax modules for tokenizing the task.
            The output of each tokenizer is concatenated to form the task token prefix.
        vocab_size (int): Number of bins for each action dimension.
        token_embedding_size (int): Size of the tokens.
        window_size (int): Length of the observation-action sequence that the transformer processes.
        pred_horizon (int): Number of actions to predict at once (the "action chunk").
        action_dim (int): Dimension of the actions.
        normalization_type (str): The type of normalization used on the actions. Either "normal" for
            normalization to unit Gaussian or "bounds" for normalization to [-1, 1].
        num_layers (int): Number of layers in transformer.
        mlp_dim (int): MLP dim of transformer.
        num_heads (int): Number of attention heads in transformer.
        dropout_rate (float): Dropout rate for transformer
        attention_dropout_rate (float): Dropout in self-attention."""

    observation_tokenizers: Sequence[nn.Module]
    task_tokenizers: Sequence[nn.Module]
    window_size: int = 1
    # Forwarded to action head
    vocab_size: int = 256
    pred_horizon: int = 1
    action_dim: int = 7
    normalization_type: str = "bounds"
    # Forwarded to Transformer
    token_embedding_size: int = 512
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
        )

        # If pred_horizon>1, we don't use the timesteps in the trajectory window where
        # we can't predict a full action chunk (i.e the last (pred_horizon-1) timesteps)
        self.horizon = self.window_size - self.pred_horizon + 1

        self.action_head = DiscretizedActionHead(
            horizon=self.horizon,
            token_embedding_size=self.token_embedding_size,
            window_size=self.window_size,
            pred_horizon=self.pred_horizon,
            action_dim=self.action_dim,
            vocab_size=self.vocab_size,
            normalization_type=self.normalization_type,
        )  # TODO: should we make action_head an argument?

        self.tokens_per_action = self.action_head.num_tokens
        self.tokens_per_obs = sum(tok.num_tokens for tok in self.observation_tokenizers)
        self.tokens_per_task = sum(tok.num_tokens for tok in self.task_tokenizers)
        self.tokens_per_time_step = self.tokens_per_obs + self.tokens_per_action

        self.total_tokens = (
            self.tokens_per_task + self.tokens_per_time_step * self.horizon
        )
        self.obs_proj = [
            nn.Dense(self.token_embedding_size)
            for _ in range(len(self.observation_tokenizers))
        ]
        self.task_proj = [
            nn.Dense(self.token_embedding_size)
            for _ in range(len(self.task_tokenizers))
        ]

        # default attention mask is causal and additionally masks previous
        # action tokens (since attending to previous actions can hurt perfomance)
        self.default_attention_mask = self.generate_default_attention_mask()

    def __call__(
        self,
        observations,
        tasks,
        actions,
        train: bool = False,
    ):
        """
        Performs a forward pass of the network and computes the loss.

        Args:
            observations: A dictionary containing observation data for a batch of trajectory windows.
                Each entry has shape (batch, window_size, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            actions: The actions in the trajectory windows with shape (batch, window_size, action_dim).
            train: Whether this is a training call.
        Returns:
            The loss, mean squared error, and accuracy for the forward pass.
        """

        assert all(
            [observations[k].shape[1] == self.window_size for k in observations]
        ), "Data loader must return trajectory windows of length window_size"

        # only use first horizon timesteps from the window
        observations = jax.tree_map(lambda x: x[:, : self.horizon], observations)

        # output is (batch, total_tokens, token_embedding_size)
        output = self.transformer_call(
            observations,
            tasks,
            train=train,
        )

        # Extract the action embeddings from the transformer output
        # and passes to the action head for loss computation
        action_embedding = self.extract_action_embeddings(output)
        loss, metrics = self.action_head.loss(
            action_embedding, actions, observations["pad_mask"]
        )
        return {**metrics, "loss": loss}

    def transformer_call(
        self,
        observations,
        tasks,
        train: bool = False,
    ):
        # combine the default attention mask and a padding mask specifc to this batch
        # to avoid attending to padding tokens
        # attention_mask is broadcast to (batch, num_heads, total_tokens, total_tokens)
        attention_mask = jnp.logical_and(
            self.default_attention_mask,
            self.generate_pad_attention_mask(observations["pad_mask"]),
        )
        task_tokens, obs_tokens, action_tokens = self.get_tokens(
            observations, tasks, train=train
        )
        input_tokens = self.assemble_input_tokens(
            task_tokens, obs_tokens, action_tokens
        )
        return self.transformer(input_tokens, attention_mask, train=train)

    def predict_action(
        self,
        observations,
        tasks,
        train: bool = False,
        argmax: bool = False,
        sample_shape: tuple = (),
        rng: PRNGKey = None,
        temperature: float = 1.0,
    ):
        """
        Predicts actions at test time.

        Args:
            observations: A dictionary containing observation data for a batch of trajectory windows.
                Each entry has shape (batch, horizon, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            train: Whether this is a training call.
            argmax: Whether to randomly sample action distribution or take the mode.
            sample_shape: The shape of samples drawn from the action distribution for visualization.
            rng: A random key for sampling the action distribution.
            temperature: The temperature to use when sampling the action distribution.
        Returns:
            The predicted actions given the provided observation history and task.
            shape (*sample_shape, batch, pred_horizon, action_dim)
            actions[..., i, :] is the prediction for i timesteps after the last observation.
            Use actions[..., 0, :] to get the prediction for the current timestep.
        """

        assert all(
            [observations[k].shape[1] == self.horizon for k in observations]
        ), "predict_action expects an observation history of length horizon"

        output = self.transformer_call(
            observations,
            tasks,
            train=train,
        )
        action_embedding = self.extract_action_embeddings(output)
        return self.action_head.predict_action(
            action_embedding,
            train=train,
            argmax=argmax,
            sample_shape=sample_shape,
            rng=rng,
            temperature=temperature,
        )

    def get_tokens(self, observations, tasks, train: bool = False):
        """
        Tokenize observation/action history and task (either goal image or language).
        """

        # a list of (batch, horizon, tokens_per_obs_tokenizer, token_embedding_size)
        obs_tokens = [
            proj(tok(observations, tasks, train=train))
            for tok, proj in zip(self.observation_tokenizers, self.obs_proj)
        ]
        # (batch, horizon, tokens_per_obs, token_embedding_size)
        obs_tokens = jnp.concatenate(obs_tokens, axis=-2)
        assert obs_tokens.shape[-2] == self.tokens_per_obs

        if len(self.task_tokenizers) > 0:
            # a list of (batch, tokens_per_task_tokenizer, token_embedding_size)
            task_tokens = [
                proj(tok(observations, tasks, train=train))
                for tok, proj in zip(self.task_tokenizers, self.task_proj)
            ]
            # (batch, tokens_per_task, token_embedding_size)
            task_tokens = jnp.concatenate(task_tokens, axis=-2)
            assert task_tokens.shape[-2] == self.tokens_per_task
        else:
            task_tokens = jnp.zeros(
                (obs_tokens.shape[0], self.tokens_per_task, self.token_embedding_size)
            )

        # we don't attend to past actions so set action tokens to zero
        # (batch, horizon, tokens_per_action, token_embedding_size)
        action_tokens = jnp.zeros(
            (
                *obs_tokens.shape[:2],
                self.tokens_per_action,
                self.token_embedding_size,
            )
        )

        return task_tokens, obs_tokens, action_tokens

    def assemble_input_tokens(
        self,
        task_tokens,
        obs_tokens,
        action_tokens,
    ):
        """
        Concatenate obs and action tokens.
        Fold horizon dim into token sequence dim.
        Prepend task tokens.

        Inputs:
            task_tokens: (batch, tokens_per_task, token_embedding_size)
            obs_tokens: (batch, horizon, tokens_per_obs, token_embedding_size)
            action_tokens: (batch, horizon, tokens_per_action, token_embedding_size)
        Returns:
            tokens: (batch, total_tokens, token_embedding_size)
        """

        # (batch, horizon, tokens_per_time_step, token_embedding_size)
        tokens = jnp.concatenate([obs_tokens, action_tokens], axis=2)

        # (batch, horizon * tokens_per_time_step, token_embedding_size)
        tokens = jnp.reshape(tokens, (tokens.shape[0], -1, tokens.shape[-1]))

        # (batch, tokens_per_task + horizon * tokens_per_time_step, token_embedding_size)
        tokens = jnp.concatenate([task_tokens, tokens], axis=1)
        return tokens

    def extract_action_embeddings(self, transformer_output):
        """
        Args:
            transformer_output: (batch, total_tokens, token_embedding_size)
        Returns:
            action_embeddings: (batch, horizon, self.tokens_per_action, token_embedding_size)
        """

        embeddings = transformer_output[:, self.tokens_per_task :]
        embeddings = jnp.reshape(
            embeddings,
            (
                embeddings.shape[0],
                self.horizon,
                self.tokens_per_time_step,
                self.token_embedding_size,
            ),
        )
        action_embeddings = embeddings[:, :, self.tokens_per_obs :]
        return action_embeddings

    def _get_token_description(self, i: int):
        """Description of what token i in the transformer is
        Args: i: index of token in transformer (0 <= i < total_tokens)
        Returns: (token_type, token_timestep, extra_metadata)
        """

        # Is it a task token?
        if i < self.tokens_per_task:
            return ("task", None, i)

        i = i - self.tokens_per_task
        timestep, position = divmod(i, self.tokens_per_time_step)

        # Observation token
        if position < self.tokens_per_obs:
            return ("obs", timestep, position)

        # Action token
        elif position < self.tokens_per_obs + self.tokens_per_action:  # Action token
            return (
                "action",
                timestep,
                self.action_head.token_metadata(position),
            )
        else:  # Value tokens coming soon?
            raise NotImplementedError()

    def generate_default_attention_mask(self):
        """
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

        attention_mask = np.zeros((self.total_tokens, self.total_tokens), dtype=int)

        for i in range(self.total_tokens):  # Token attending
            for j in range(self.total_tokens):  # Token being attended to
                # description is a tuple (token_type, token_timestep, extra_info)
                description_i = self._get_token_description(i)
                description_j = self._get_token_description(j)

                if description_i[0] == "task":
                    # Only attend to other task tokens
                    mask = 1 if description_j[0] == "task" else 0
                elif description_i[0] == "obs":
                    # Only attend to observation tokens in the same timestep or before
                    if description_j[0] == "task":
                        mask = 1
                    elif description_j[0] == "obs":
                        mask = 1 if description_j[1] <= description_i[1] else 0
                    else:
                        mask = 0  # Don't attend to actions
                else:
                    mask = self.action_head.attention_mask_ij(
                        description_i, description_j
                    )
                attention_mask[i, j] = mask

        return attention_mask

    def generate_pad_attention_mask(self, pad_mask):
        """
        Generate attention mask that ignores padding. `pad_mask` has shape (batch, horizon) and
        records which time steps are padding. We first expand the mask to shape (batch, horizon * tokens_per_time_step)
        and then prepend a mask for the task prefix to get shape (batch, total_tokens).
        We broadcast to (batch, num_heads, total_tokens, total_tokens).
        """

        sequence_mask = jnp.repeat(pad_mask, self.tokens_per_time_step, axis=1)
        task_mask = jnp.ones((pad_mask.shape[0], self.tokens_per_task), dtype=int)
        full_mask = jnp.concatenate([task_mask, sequence_mask], axis=1)
        full_mask = jnp.broadcast_to(
            full_mask[:, None, None, :],
            (
                full_mask.shape[0],
                self.num_heads,
                self.total_tokens,
                self.total_tokens,
            ),
        )
        return full_mask
