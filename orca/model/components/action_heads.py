# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
from typing import Dict, Optional

import distrax
from einops import rearrange
import flax.linen as nn
import jax
import jax.numpy as jnp

from orca.model.components.base import TokenGroup
from orca.model.components.tokenizers import BinTokenizer
from orca.model.components.transformer import MAPHead
from orca.utils.typing import PRNGKey


def masked_mean(x, mask):
    mask = jnp.broadcast_to(mask, x.shape)
    return jnp.mean(x * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)


def chunk_actions(actions, pred_horizon):
    """Chunk actions for predicting actions `pred_horizon` steps into the future.

    The resulting actions have shape (batch, actions.shape[-2] - (pred_horizon - 1), pred_horizon, action_dim)

    This folds chunk_actions([a_1, a_2, a_3, a_4, a_5], 3) into:
     [
       [a_1, a_2, a_3],
       [a_2, a_3, a_4],
       [a_3, a_4, a_5],
    ]

    """
    window_size = actions.shape[1]
    assert window_size >= pred_horizon, "Chunk size too large for action window size"
    chunk_window_size = window_size - (pred_horizon - 1)

    curr_step = jnp.arange(chunk_window_size)
    action_offset = jnp.arange(pred_horizon)
    chunk_indices = curr_step[:, None] + action_offset[None, :]
    return actions[:, chunk_indices]


def continuous_loss(pred_value, ground_truth_value, mask, loss_type="mse"):
    """
    Args:
        pred_value: jnp.ndarray w/ shape (batch_dims...)
        ground_truth_value: continuous values in jnp.ndarray w/ shape (batch_dims...)
        mask: jnp.ndarray broadcastable to ground_truth
    """
    if loss_type == "mse":
        loss = jnp.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = jnp.abs(pred_value - ground_truth_value)

    loss = masked_mean(loss, mask)

    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
    }


def discrete_loss(discrete_tokenizer, logits, ground_truth_value, mask):
    """
    Args:
        discrete_tokenizer: BinTokenizer
        logits: jnp.ndarray w/ shape (batch_dims..., vocab_size)
        ground_truth_value: continuous values in jnp.ndarray w/ shape (batch_dims...)
        mask: jnp.ndarray broadcastable to ground_truth
    """
    labels = discrete_tokenizer(ground_truth_value)
    labels_one_hot = jax.nn.one_hot(labels, logits.shape[-1])

    loss = -jnp.sum(logits * labels_one_hot, axis=-1)
    loss = masked_mean(loss, mask)

    # compute accuracy between predicted actions and target actions
    pred_label = jnp.argmax(logits, axis=-1)
    accuracy = pred_label == labels
    accuracy = masked_mean(accuracy, mask)

    # detokenize the predicted actions
    pred_value = discrete_tokenizer.decode(pred_label)
    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
        "accuracy": accuracy,
    }


class ContinuousActionHead(nn.Module):
    """Predicts continuous actions instead of discretized actions.

    Continuous actions are predicted, tanh squashed to [-max_action, max_action],
    and then regressed using MSE error.

    You may create an embedding by either mean-pooling across
    tokens or using multi-head attention pooling (use_map). It is recommended
    to use MAP when decoding from the observation token stream.
    """

    readout_key: str
    use_map: bool
    pred_horizon: int = 1
    action_dim: int = 7
    max_action: float = 5.0
    loss_type: str = "mse"

    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()
        self.mean_proj = nn.Dense(self.pred_horizon * self.action_dim)

    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train=True
    ) -> jax.Array:
        """
        Args:
            embeddings: jnp.ndarray w/ shape (batch_size, horizon, n_tokens, embedding_size)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, horizon, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:
            embeddings = token_group.tokens.mean(axis=-2)
        # embeddings is (batch_size, horizon, embedding_size)

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

        actions_chunked = chunk_actions(actions, self.pred_horizon)
        actions_chunked = actions_chunked[:, :horizon]

        loss, metrics = continuous_loss(
            mean, actions_chunked, pad_mask, loss_type="mse"
        )
        # Sum over action dimension instead of averaging
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        argmax: bool = False,
        sample_shape: tuple = (),
        rng: Optional[PRNGKey] = None,
        temperature: float = 1.0,
    ) -> jax.Array:
        # Outputs the predicted actions for the final
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
    readout_key: Optional[str] = None
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

    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train=True
    ) -> jax.Array:
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

            Did you make sure to set "future_action_window_size" correctly in the data config?
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
        rng: Optional[PRNGKey] = None,
        temperature: float = 1.0,
    ) -> jax.Array:
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

    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train=True
    ) -> jax.Array:
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

    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train=True
    ) -> jax.Array:
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

    def _objective(self, actions, pred_actions, pad_mask):
        action_mse = jnp.square(actions - pred_actions).sum(axis=-1)
        return masked_mean(action_mse, pad_mask[:, :, None])

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

            Did you make sure to set "future_action_window_size" correctly in the data config?
        """
        # chunk the target actions to match the predicted actions
        # only use first horizon timesteps from the window
        actions = jnp.clip(actions, -self.max_action, self.max_action)
        actions_chunked = self._chunk_actions(actions)
        horizon = mean.shape[1]
        actions_chunked = actions_chunked[:, :horizon]

        action_loss = self._objective(actions_chunked, mean, pad_mask)

        # log mse and mse histogram
        action_mse = jnp.square(actions_chunked - mean).sum(axis=-1)
        action_mse_hist = (action_mse * pad_mask[:, :, None]).reshape(-1)
        action_mse = masked_mean(action_mse, pad_mask[:, :, None])

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
        rng: Optional[PRNGKey] = None,
        temperature: float = 1.0,
    ) -> jax.Array:
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


class L1ActionHead(MSEActionHead):
    def _objective(self, actions, pred_actions, pad_mask):
        action_l1 = jnp.abs(actions - pred_actions).sum(axis=-1)
        return masked_mean(action_l1, pad_mask[:, :, None])


ACTION_HEADS = {
    "basic_action_head": BasicActionHead,
    "token_per_dim_action_head": TokenPerDimActionHead,
    "mse_action_head": MSEActionHead,
    "l1_action_head": L1ActionHead,
}
