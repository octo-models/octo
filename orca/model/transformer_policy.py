# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.tokenizers import ActionTokenizer
from orca.model.transformer import Transformer
from orca.typing import PRNGKey, Sequence


class TransformerPolicy(nn.Module):
    """
    This transformer models a sequence of observations and actions (of length `horizon`),
    prefixed by a task (either a goal image or language instruction). The observations,
    actions, and task are tokenized and then assembled into a token sequence.

    With no action chunking (`pred_horizon=1`) the token sequence looks like:

    [task, o_0, a_0, o_1, a_1, ..., o_n, a_n]

    With action chunking (`pred_horizon>1`), the actions in the chunk are packed between
    the observations. For example, if `pred_horizon=3`, the sequence looks like:

    [task, o_0, a_0, a_1, a_2, o_1, a_1, a_2, a_3, o_2, ..., o_n, a_n, a_n+1, a_n+2]

    In both cases, a causal mask ensures that each token can only attend to past tokens.

    At test time, we predict actions non-autoregressively (i.e we predict all the tokens
    of all the actions in a chunk in one forward pass). We do this by using mask/padding tokens
    in the input sequence in place of the action tokens. Thus, all the action tokens are masked
    at training time as well. Note: even though the attention mask ensures we don't attend to the
    actions, we still need to mask them because of the residual connection in the attention block.
    When rolling out the policy, we left pad the observation-action history until we have enough
    history to fill the entire context window.

    Parameters:
        observations_tokenizers: List of flax modules for tokenizing the observations. The output
            of each tokenizer is concatenated to form the observation tokens.
        task_tokenizers: List of flax modules for tokenizing the task. The output of each tokenizer
            is concatenated to form the task token prefix.
        vocab_size: Number of bins for each action dimension.
        token_embedding_size: Size of the tokens.
        horizon: Length of the observation-action sequence that the transformer processes.
        pred_horizon: Number of actions to predict at once (the "action chunk").
        action_dim: Dimension of the actions.
        normalization_type: The type of normalization used on the actions. Either "normal" for
            normalization to unit Gaussian or "bounds" for normalization to [-1, 1].
        num_layers: Number of layers in transformer.
        mlp_dim: MLP dim of transformer.
        num_heads: Number of heads in transformer.
        dropout_rate: Dropout rate for transformer
        attention_dropout_rate: Dropout in self-attention."""

    observation_tokenizers: Sequence[nn.Module]
    task_tokenizers: Sequence[nn.Module]
    vocab_size: int = 256
    token_embedding_size: int = 512
    window_size: int = 1
    pred_horizon: int = 1
    action_dim: int = 7
    normalization_type: str = "bounds"
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

        self.action_tokenizer = ActionTokenizer(
            action_dim=self.action_dim,
            vocab_size=self.vocab_size,
            normalization_type=self.normalization_type,
        )

        assert (
            self.window_size >= self.pred_horizon
        ), "Trajectory must contain enough actions to predict a full chunk."
        # If pred_horizon>1, we don't use the timesteps in the trajectory window where
        # we can't predict a full action chunk (i.e the last (pred_horizon-1) timesteps)
        self.horizon = self.window_size - self.pred_horizon + 1

        self.tokens_per_action = self.action_dim * self.pred_horizon
        self.tokens_per_obs = sum(tok.num_tokens for tok in self.observation_tokenizers)
        self.tokens_per_task = sum(tok.num_tokens for tok in self.task_tokenizers)
        self.tokens_per_time_step = self.tokens_per_obs + self.tokens_per_action
        self.total_tokens = (
            self.tokens_per_task + self.tokens_per_time_step * self.horizon
        )
        self.vocab_proj = nn.Dense(self.vocab_size)
        self.obs_proj = [
            nn.Dense(self.token_embedding_size)
            for _ in range(len(self.observation_tokenizers))
        ]
        self.task_proj = [
            nn.Dense(self.token_embedding_size)
            for _ in range(len(self.task_tokenizers))
        ]

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

        # only use first horizon timesteps from the window
        observations = jax.tree_map(lambda x: x[:, : self.horizon], observations)

        # output is (batch, horizon, tokens_per_time_step, token_embedding_size)
        output = self.transformer_call(
            observations,
            tasks,
            train=train,
        )

        # get the logits for all the actions by taking the action tokens of each timestep,
        # unfolding the pred_horizon dim, and projecting to the vocab size
        # (batch, horizon, tokens_per_action, token_embedding_size)
        action_embedding = output[:, :, -self.tokens_per_action :]
        # (batch, horizon, pred_horizon, action_dim, token_embedding_size)
        action_embedding = jnp.reshape(
            action_embedding,
            (
                *action_embedding.shape[:2],
                self.pred_horizon,
                self.action_dim,
                self.token_embedding_size,
            ),
        )
        # (batch, horizon, pred_horizon, action_dim, vocab_size)
        action_logits = self.vocab_proj(action_embedding)

        action_logprob = jax.nn.log_softmax(action_logits, axis=-1)

        actions_chunked = self.chunk_actions(actions)

        # only use first horizon timesteps from the window
        actions_chunked = actions_chunked[:, : self.horizon]

        action_labels = self.action_tokenizer(actions_chunked, mode="tokenize")
        action_labels_one_hot = jax.nn.one_hot(action_labels, self.vocab_size)

        action_loss = -jnp.sum(action_logprob * action_labels_one_hot, axis=-1)
        action_loss = (action_loss * observations["pad_mask"][:, :, None, None]).mean()

        action_pred = jnp.argmax(action_logits, axis=-1)
        accuracy = action_pred == action_labels
        accuracy = (accuracy * observations["pad_mask"][:, :, None, None]).mean()

        action_values = self.action_tokenizer(action_pred, mode="detokenize")
        action_mse = jnp.square(actions_chunked - action_values).sum(axis=-1)
        action_mse = (action_mse * observations["pad_mask"][:, :, None]).mean()

        return {"loss": action_loss, "mse": action_mse, "accuracy": accuracy}

    def transformer_call(
        self,
        observations,
        tasks,
        train: bool = False,
    ):
        # combine the default attention mask and a padding mask specifc to this batch
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
        output = self.transformer(input_tokens, attention_mask, train=train)
        # remove output corresponding to task
        output = output[:, self.tokens_per_task :]

        # unfold horizon length from token sequence length
        return jnp.reshape(
            output,
            (
                output.shape[0],
                self.horizon,
                self.tokens_per_time_step,
                self.token_embedding_size,
            ),
        )

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
                Each entry has shape (batch, window_size, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            train: Whether this is a training call.
            argmax: Whether to randomly sample action distribution or take the mode.
            sample_shape: The shape of samples drawn from the action distribution for visualization.
            rng: A random key for sampling the action distribution.
            temperature: The temperature to use when sampling the action distribution.
        Returns:
            The predicted actions given the provided observation history and task.
        """

        # only use first horizon timesteps from the window
        observations = jax.tree_map(lambda x: x[:, : self.horizon], observations)

        output = self.transformer_call(
            observations,
            tasks,
            train=train,
        )

        # get the logits for the last action by taking the action tokens of the last timestep,
        # unfolding the pred_horizon dim, and projecting to the vocab size
        # (batch, tokens_per_action, token_embedding_size)
        action_embedding = output[:, -1, -self.tokens_per_action :]
        # (batch, pred_horizon, action_dim, token_embedding_size)
        action_embedding = jnp.reshape(
            action_embedding,
            (
                action_embedding.shape[0],
                self.pred_horizon,
                self.action_dim,
                self.token_embedding_size,
            ),
        )
        # (batch, pred_horizon, action_dim, vocab_size)
        action_logits = self.vocab_proj(action_embedding)

        if argmax:
            action_tokens = jnp.argmax(action_logits, axis=-1).astype(jnp.int32)
        else:
            dist = distrax.Categorical(logits=action_logits / temperature)
            action_tokens = dist.sample(seed=rng, sample_shape=sample_shape).astype(
                jnp.int32
            )
        return self.action_tokenizer(action_tokens, mode="detokenize")

    def generate_default_attention_mask(self):
        """
        Generate default attention mask for transformer call.
        """

        attention_mask = np.zeros((self.total_tokens, self.total_tokens), dtype=int)

        # mask for obs-action sequence
        for i in range(self.tokens_per_time_step * self.horizon):
            for j in range(self.tokens_per_time_step * self.horizon):
                # get the index in the time sequence given the token index
                index_i = int(i / self.tokens_per_time_step)
                index_j = int(j / self.tokens_per_time_step)
                # determine whether this token represents an action or observation
                is_action_j = j % self.tokens_per_time_step >= self.tokens_per_obs

                mask = 1
                # don't attend to future timesteps
                if index_j > index_i:
                    mask = 0
                # don't attend to actions
                if is_action_j:
                    mask = 0
                attention_mask[
                    self.tokens_per_task + i, self.tokens_per_task + j
                ] = mask

        # add mask for task tokens
        attention_mask[:, : self.tokens_per_task] = 1

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
        """

        # (batch, horizon, tokens_per_time_step, token_embedding_size)
        tokens = jnp.concatenate([obs_tokens, action_tokens], axis=2)

        # (batch, horizon * tokens_per_time_step, token_embedding_size)
        tokens = jnp.reshape(tokens, (tokens.shape[0], -1, tokens.shape[-1]))

        # (batch, tokens_per_task + horizon * tokens_per_time_step, token_embedding_size)
        tokens = jnp.concatenate([task_tokens, tokens], axis=1)
        return tokens

    def chunk_actions(self, actions):
        """
        Chunk actions into `pred_horizon` size chunks.
        The resulting actions have shape (batch, window_size, pred_horizon, action_dim)
        """

        chunk_indices = jnp.broadcast_to(
            jnp.arange(self.pred_horizon), [self.window_size, self.pred_horizon]
        ) + jnp.broadcast_to(
            jnp.arange(self.window_size)[:, None],
            [self.window_size, self.pred_horizon],
        )
        chunk_indices = jnp.minimum(chunk_indices, self.window_size - 1)
        return actions[:, chunk_indices]
