# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
from typing import Any

import distrax
from einops import rearrange
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.components.base import TokenGroup
from orca.model.components.tokenizers import BinTokenizer
from orca.model.components.transformer import MAPHead
from orca.utils.typing import Dict, PRNGKey


def masked_mean(x, mask):
    mask = jnp.broadcast_to(mask, x.shape)
    return jnp.mean(x * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)


class BasicActionHead(nn.Module):
    """
    A basic action decoding head that predicts discretized actions using
    an arbitrary group of token embeddings.

    Pools all the token embeddings for a timestep together (either via mean pooling
    or via multi-head attention pooling (specified by `use_map`). Then, predictins
    logits for each action dimension and prediction horizon simultaneously
    using a linear projection on a shared embedding.

    Supports predicting for multiple timesteps at once.
    """

    pred_horizon: int = 1
    action_dim: int = 7
    vocab_size: int = 256
    normalization_type: str = "bounds"
    readout_key: str = None
    use_map: bool = False

    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()

        self.vocab_proj = nn.Dense(
            self.vocab_size * self.pred_horizon * self.action_dim,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.constant(-1 * jnp.log(self.vocab_size)),
        )
        self.action_tokenizer = BinTokenizer(
            n_bins=self.vocab_size,
            bin_type="uniform"
            if self.normalization_type == "bounds"
            else self.normalization_type,
        )

    def __call__(self, transformer_outputs: Dict[str, TokenGroup], train=True) -> Any:
        """
        Args:
            transformer_outputs: Dict[str, TokenGroup] the output of an OrcaTransformer
        """
        assert self.readout_key is not None
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, horizon, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, horizon, embedding_size)

        # (batch_size, horizon,  pred_horizon * action_dim * vocab_size)
        logits = self.vocab_proj(embeddings)
        logits = rearrange(
            logits,
            "b h (p a d) -> b h p a d",
            p=self.pred_horizon,
            a=self.action_dim,
            d=self.vocab_size,
        )
        return logits

    def loss(
        self, transformer_outputs: Dict[str, TokenGroup], actions, pad_mask, train=True
    ):
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
        action_logits = self.__call__(transformer_outputs, train=train)

        horizon = action_logits.shape[1]
        assert (
            actions.shape[1] >= horizon + self.pred_horizon - 1
        ), f"""
            To predict actions for horizon {horizon} and future prediction horizon {self.pred_horizon},
            the ground-truth actions must have at least {horizon + self.pred_horizon - 1} timesteps, but got shape {actions.shape}.

            Did you make sure to set "additional_action_window_size" correctly in the data config?
        """

        # compute log probabilities for predicted actions
        action_logprob = jax.nn.log_softmax(action_logits, axis=-1)

        # chunk the target actions to match the predicted actions
        actions_chunked = self._chunk_actions(actions)

        # only use first horizon timesteps from the window
        horizon = action_logprob.shape[1]
        actions_chunked = actions_chunked[:, :horizon]

        # tokenize the target actions and convert them to one hot vectors
        action_labels = self.action_tokenizer(actions_chunked)
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
        action_values = self.action_tokenizer.decode(action_pred)
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
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        argmax: bool = False,
        sample_shape: tuple = (),
        rng: PRNGKey = None,
        temperature: float = 1.0,
    ):
        # get the logits for the last action by taking the action tokens of the last timestep,
        # unfolding the pred_horizon dim, and projecting to the vocab size
        # (batch, tokens_per_action, token_embedding_size)

        action_logits = self.__call__(transformer_outputs, train=train)
        action_logits = action_logits[:, -1]

        if argmax:
            action_tokens = jnp.argmax(action_logits, axis=-1).astype(jnp.int32)
        else:
            dist = distrax.Categorical(logits=action_logits / temperature)
            action_tokens = dist.sample(seed=rng, sample_shape=sample_shape).astype(
                jnp.int32
            )
        return self.action_tokenizer.decode(action_tokens)

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
        if self.use_map:
            self.map_head = MAPHead(num_readouts=self.pred_horizon * self.action_dim)

        self.vocab_proj = nn.Dense(
            self.vocab_size,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.constant(-1 * jnp.log(self.vocab_size)),
        )
        self.action_tokenizer = BinTokenizer(
            n_bins=self.vocab_size,
            bin_type="uniform"
            if self.normalization_type == "bounds"
            else self.normalization_type,
        )

    def __call__(self, transformer_outputs: Dict[str, TokenGroup], train=True) -> Any:
        """
        Args:
            embeddings: jnp.ndarray w/ shape (batch_size, horizon, n_tokens, embedding_size)
        """
        assert self.readout_key is not None
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, horizon, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:
            embeddings = self.map_head(token_group, train=train)
        else:
            embeddings = token_group.tokens
            assert embeddings.shape[-2] == self.pred_horizon * self.action_dim

        # Now, embeddings is (batch_size, horizon, pred_horizon * action_dim, embedding_size)
        logits = self.vocab_proj(embeddings)
        logits = rearrange(
            logits, "b h (p a) d -> b h p a d", p=self.pred_horizon, a=self.action_dim
        )
        return logits


class MSEActionHead(BasicActionHead):
    """Predicts continuous actions instead of discretized actions.

    Continuous actions are predicted, tanh squashed to [-max_action, max_action],
    and then regressed using MSE error.

    As in BasicActionHead, you may create an embedding by either mean-pooling across
    tokens or using multi-head attention pooling (use_map). It is recommended
    when decoding from the observation token stream to use MAP.
    """

    max_action: float = 5.0  # Handles OOD actions during training / eval
    use_map: bool = True

    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()
        self.mean_proj = nn.Dense(self.pred_horizon * self.action_dim)

    def __call__(self, transformer_outputs: Dict[str, TokenGroup], train=True) -> Any:
        """
        Args:
            embeddings: jnp.ndarray w/ shape (batch_size, horizon, n_tokens, embedding_size)
        """
        assert self.readout_key is not None
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, horizon, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, horizon, embedding_size)
        mean = self.mean_proj(embeddings)
        mean = rearrange(
            mean, "b h (p a) -> b h p a", p=self.pred_horizon, a=self.action_dim
        )
        mean = jnp.tanh(mean / self.max_action) * self.max_action
        return mean

    def loss(
        self, transformer_outputs: Dict[str, TokenGroup], actions, pad_mask, train=True
    ):
        """
        Trains the mean head with MSE and the logstd head with KL divergence.

        Args:
            embeddings: jnp.ndarray w/ shape (batch_size, horizon, num_tokens, embedding_size)
            actions: jnp.ndarray w/ shape (batch_size, >= horizon + pred_horizon - 1, action_dim)
            pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep.

        Returns:
            loss: float
            metrics: dict
        """
        # (batch, horizon, pred_horizon, action_dim)
        mean = self.__call__(transformer_outputs, train=train)

        horizon = mean.shape[1]
        assert (
            actions.shape[1] >= horizon + self.pred_horizon - 1
        ), f"""
            To predict actions for horizon {horizon} and future prediction horizon {self.pred_horizon},
            the ground-truth actions must have at least {horizon + self.pred_horizon - 1} timesteps, but got shape {actions.shape}.

            Did you make sure to set "additional_action_window_size" correctly in the data config?
        """
        # chunk the target actions to match the predicted actions
        # only use first horizon timesteps from the window
        actions = jnp.clip(actions, -self.max_action, self.max_action)
        actions_chunked = self._chunk_actions(actions)
        horizon = mean.shape[1]
        actions_chunked = actions_chunked[:, :horizon]

        action_mse = jnp.square(actions_chunked - mean).sum(axis=-1)
        action_mse_hist = (action_mse * pad_mask[:, :, None]).reshape(-1)
        action_mse = masked_mean(action_mse, pad_mask[:, :, None])

        action_loss = action_mse

        return action_loss, {
            "loss": action_loss,
            "mse": action_mse,
            "mse_hist": action_mse_hist,
        }

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        argmax: bool = False,
        sample_shape: tuple = (),
        rng: PRNGKey = None,
        temperature: float = 1.0,
    ):
        # get the logits for the last action by taking the action tokens of the last timestep,
        # unfolding the pred_horizon dim, and projecting to the vocab size
        # (batch, tokens_per_action, token_embedding_size)
        mean = self.__call__(transformer_outputs, train=train)
        mean = mean[:, -1]
        logstd = jnp.full_like(mean, -10.0)

        if argmax:
            action = mean
        else:
            dist = distrax.MultivariateNormalDiag(mean, jnp.exp(logstd) * temperature)
            action = dist.sample(seed=rng, sample_shape=sample_shape)
        return action


ACTION_HEADS = {
    "basic_action_head": BasicActionHead,
    "token_per_dim_action_head": TokenPerDimActionHead,
    "mse_action_head": MSEActionHead,
}
