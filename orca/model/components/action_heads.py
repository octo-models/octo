# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
from typing import Any

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.components.tokenizers import ActionTokenizer
from orca.utils.typing import PRNGKey


class BasicActionHead(nn.Module):
    """
    A basic action decoding head that predicts discretized actions using
    an arbitrary group of token embeddings.

    Pools all the token embeddings for a timestep together before predicting
    logits for each action dimension and prediction horizon simultaneously
    using a linear projection on a shared embedding.

    Supports predicting for multiple timesteps at once.
    """

    pred_horizon: int = 1
    action_dim: int = 7
    vocab_size: int = 256
    normalization_type: str = "bounds"

    def setup(self):
        self.vocab_proj = nn.Dense(
            self.vocab_size * self.pred_horizon * self.action_dim
        )
        self.action_tokenizer = ActionTokenizer(
            vocab_size=self.vocab_size,
            normalization_type=self.normalization_type,
        )

    def __call__(self, embeddings, train=True) -> Any:
        """
        Args:
            embeddings: jnp.ndarray w/ shape (batch_size, horizon, n_tokens, embedding_size)
        """
        batch_size, horizon, n_tokens, embedding_size = embeddings.shape

        embeddings = embeddings.mean(
            axis=-2
        )  # Now, embeddings is (batch_size, horizon, embedding_size)
        logits = self.vocab_proj(
            embeddings
        )  # (batch_size, horizon, vocab_size * pred_horizon * action_dim)
        logits = jnp.reshape(
            logits,
            (
                batch_size,
                horizon,
                self.pred_horizon,
                self.action_dim,
                self.vocab_size,
            ),
        )
        return logits

    def loss(self, embeddings, actions, pad_mask, train=True):
        """
        Args:
            embeddings: jnp.ndarray w/ shape (batch_size, horizon, num_tokens, embedding_size)
            actions: jnp.ndarray w/ shape (batch_size, >= horizon + pred_horizon - 1, action_dim)
            pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep.

        Returns:
            loss: float
            metrics: dict
        """

        # get the logits for all the actions by taking the action tokens of each timestep,
        # unfolding the pred_horizon dim, and projecting to the vocab size

        # (batch, horizon, pred_horizon, action_dim, token_embedding_size)
        action_logits = self.__call__(embeddings, train=train)
        # compute log probabilities for predicted actions
        action_logprob = jax.nn.log_softmax(action_logits, axis=-1)

        # chunk the target actions to match the predicted actions
        actions_chunked = self._chunk_actions(actions)

        # only use first horizon timesteps from the window
        horizon = actions_chunked.shape[1]
        actions_chunked = actions_chunked[:, :horizon]

        # tokenize the target actions and convert them to one hot vectors
        action_labels = self.action_tokenizer(actions_chunked, mode="tokenize")
        action_labels_one_hot = jax.nn.one_hot(action_labels, self.vocab_size)

        # compute the CE loss using the log probabilities and target actions
        action_loss = -jnp.sum(action_logprob * action_labels_one_hot, axis=-1)
        # mask the loss with the pad mask to avoid supervising padding
        action_loss = (action_loss * pad_mask[:, :, None, None]).mean()

        # take the highest probability actions as the predicted actions
        action_pred = jnp.argmax(action_logits, axis=-1)

        # compute accuracy between predicted actions and target actions
        accuracy = action_pred == action_labels
        # mask the accuracy with the pad mask to remove the contribution of padding
        accuracy = (accuracy * pad_mask[:, :, None, None]).mean()

        # detokenize the predicted actions
        action_values = self.action_tokenizer(action_pred, mode="detokenize")
        # compute the mean squared error between predicted actions and target actions
        action_mse = jnp.square(actions_chunked - action_values).sum(axis=-1)
        # mask the mse with the pad mask to remove the contribution of padding
        action_mse = (action_mse * pad_mask[:, :, None]).mean()

        return action_loss, {
            "loss": action_loss,
            "mse": action_mse,
            "accuracy": accuracy,
        }

    def predict_action(
        self,
        embeddings,
        train: bool = True,
        argmax: bool = False,
        sample_shape: tuple = (),
        rng: PRNGKey = None,
        temperature: float = 1.0,
    ):
        # get the logits for the last action by taking the action tokens of the last timestep,
        # unfolding the pred_horizon dim, and projecting to the vocab size
        # (batch, tokens_per_action, token_embedding_size)

        action_logits = self.__call__(embeddings, train=train) * temperature
        action_logits = action_logits[:, -1]

        if argmax:
            action_tokens = jnp.argmax(action_logits, axis=-1).astype(jnp.int32)
        else:
            dist = distrax.Categorical(logits=action_logits / temperature)
            action_tokens = dist.sample(seed=rng, sample_shape=sample_shape).astype(
                jnp.int32
            )
        return self.action_tokenizer(action_tokens, mode="detokenize")

    def _chunk_actions(self, actions):
        """
        Chunk actions into `pred_horizon` size chunks.
        The resulting actions have shape (batch, window_size, pred_horizon, action_dim)
        """
        window_size = actions.shape[1]
        chunk_indices = jnp.broadcast_to(
            jnp.arange(self.pred_horizon), [window_size, self.pred_horizon]
        ) + jnp.broadcast_to(
            jnp.arange(window_size)[:, None],
            [window_size, self.pred_horizon],
        )
        chunk_indices = jnp.minimum(chunk_indices, window_size - 1)
        return actions[:, chunk_indices]


class TokenPerDimActionHead(BasicActionHead):
    """
    Assumes that there is a separate embedding for each dimension of the action
    and each future prediction horizon.

    E.g. that the input embedding has shape (batch_size, horizon, pred_horizon * action_dim, embedding_size)

    Supports predicting for multiple timesteps at once
    """

    def setup(self):
        self.vocab_proj = nn.Dense(self.vocab_size)
        self.action_tokenizer = ActionTokenizer(
            vocab_size=self.vocab_size,
            normalization_type=self.normalization_type,
        )

    def __call__(self, embeddings, train=True) -> Any:
        """
        Args:
            embeddings: jnp.ndarray w/ shape (batch_size, horizon, n_tokens, embedding_size)
        """
        batch_size, horizon, n_tokens, embedding_size = embeddings.shape
        assert n_tokens == self.pred_horizon * self.action_dim
        # (batch_size, horizon, pred_horizon * action_dim, vocab_size)
        logits = self.vocab_proj(embeddings)
        logits = jnp.reshape(
            logits,
            (
                batch_size,
                horizon,
                self.pred_horizon,
                self.action_dim,
                self.vocab_size,
            ),
        )
        return logits


ACTION_HEADS = {
    "basic_action_head": BasicActionHead,
    "token_per_dim_action_head": TokenPerDimActionHead,
}
