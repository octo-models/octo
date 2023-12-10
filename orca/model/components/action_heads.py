"""Action prediction modules that take in the transformer token outputs and predict actions.

Each action head here does chunked action prediction: i.e. at every timestep,
it tries to predict the next `pred_horizon` actions into the future from that timestep.
Setting `pred_horizon=1` corresponds to the typical action prediction setup.

The base structure of an action head is as follows:

class ActionHead(nn.Module):
    def loss(self, transformer_outputs, actions, pad_mask, train=True):
        # Compute the loss and metrics for training the action head.
        return loss, metrics

    def predict_action(self, transformer_outputs, argmax=False, sample_shape=(), rng=None, temperature=1.0):
        # Predict the action for the most recent timestep.
        return predicted_action # jnp.ndarray w/ shape (*sample_shape, batch_size, pred_horizon, action_dim)

"""
from functools import partial
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


def discrete_loss(discrete_tokenizer: BinTokenizer, logits, ground_truth_value, mask):
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
    use_map: bool = False
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
        Returns:
            mean: Predicted actions, jnp.ndarray w/ shape (batch_size, horizon, pred_horizon, action_dim)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, horizon, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
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
        _check_action_window_size(actions, horizon, self.pred_horizon)
        actions_chunked = chunk_actions(actions, self.pred_horizon)
        actions_chunked = actions_chunked[:, :horizon]

        loss, metrics = continuous_loss(
            mean, actions_chunked, pad_mask[:, :, None, None], loss_type="mse"
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


class DiscreteActionHead(nn.Module):
    """
    A basic action decoding head that predicts discretized actions using
    the transformer token embeddings.



    Token_per determines how many tokens are used to represent each action.
        - If "" (an empty string): then a single token is responsible for producing the action logits
            for all dimensions at all future prediction horizons
        - If "pred_horizon", then we use `self.pred_horizon` tokens, each responsible for producing the action logits
            for all dimensions at the corresponding future prediction horizon
        - If "action_dim_and_pred_horizon", then we use `self.pred_horizon * self.action_dim` tokens, where
            each token is responsible for the logits for the specific dim and timestep

    If MAP is used, then the correct # of tokens are automatically created, otherwise readout_key must have exactly the right number of tokens.
    """

    readout_key: str
    use_map: bool = False
    token_per: str = "action_dim_and_pred_horizon"
    pred_horizon: int = 1
    action_dim: int = 7
    vocab_size: int = 256
    normalization_type: str = "uniform"

    def setup(self):
        total_output = self.pred_horizon * self.action_dim * self.vocab_size

        if self.token_per == "":
            self.n_tokens = 1
            self.final_layer_size = total_output
        elif self.token_per == "pred_horizon":
            self.n_tokens = self.pred_horizon
            self.final_layer_size = total_output // self.pred_horizon
        elif self.token_per == "action_dim_and_pred_horizon":
            self.n_tokens = self.pred_horizon * self.action_dim
            self.final_layer_size = self.vocab_size
        else:
            raise ValueError(f"Invalid token_per: {self.token_per}")

        if self.use_map:
            self.map_head = MAPHead(num_readouts=self.n_tokens)

        self.vocab_proj = nn.Dense(self.final_layer_size)
        self.action_tokenizer = BinTokenizer(
            n_bins=self.vocab_size,
            bin_type=self.normalization_type,
        )

    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train=True
    ) -> jax.Array:
        """
        Returns:
            logits: jnp.ndarray w/ shape (batch_size, horizon,  pred_horizon, action_dim, vocab_size)
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
            assert (
                embeddings.shape[-2] == self.n_tokens
            ), f"Discrete action head expects {self.n_tokens} tokens"

        # Now, embeddings is (batch_size, horizon, # tokens, embedding_size)
        batch_size, horizon = embeddings.shape[:2]

        logits = self.vocab_proj(embeddings)
        logits = logits.reshape(
            batch_size, horizon, self.pred_horizon, self.action_dim, self.vocab_size
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
        _check_action_window_size(actions, horizon, self.pred_horizon)

        actions_chunked = chunk_actions(actions, self.pred_horizon)
        actions_chunked = actions_chunked[:, :horizon]

        loss, metrics = discrete_loss(
            self.action_tokenizer,
            action_logits,
            actions_chunked,
            pad_mask[:, :, None, None],
        )

        # For MSE, sum over action dimension instead of averaging
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


def _check_action_window_size(actions, obs_horizon, pred_horizon):
    assert (
        actions.shape[1] >= obs_horizon + pred_horizon - 1
    ), f"""
        To predict actions for horizon {obs_horizon} and future prediction horizon {pred_horizon},
        the ground-truth actions must have at least {obs_horizon + pred_horizon - 1} timesteps, but got shape {actions.shape}.

        Did you make sure to set "future_action_window_size" correctly in the data config?
    """


class MSEActionHead(ContinuousActionHead):
    max_action: float = 5.0
    loss_type: str = "mse"
    use_map: bool = True


class L1ActionHead(ContinuousActionHead):
    max_action: float = 5.0
    loss_type: str = "l1"
    use_map: bool = True


class TokenPerDimActionHead(DiscreteActionHead):
    token_per: str = "action_dim_and_pred_horizon"


ACTION_HEADS = {
    "token_per_dim_action_head": TokenPerDimActionHead,
    "mse_action_head": MSEActionHead,
    "l1_action_head": L1ActionHead,
}
