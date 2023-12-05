# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
from typing import Optional

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from orca.model.components.tokenizers import BinTokenizer
from orca.utils.typing import PRNGKey


class BaseTemporalHead(nn.Module):
    """
    A reward/value decoding head that predicts discretized temporal distances.

    Pools all the token embeddings for a timestep together before predicting
    logits for each temporal distance bin using a linear readout.
    """

    n_bins: int = 256
    bin_type: str = "uniform"
    readout_key: Optional[str] = None

    def setup(self):
        self.readout_proj = nn.Dense(self.n_bins)
        self.distance_tokenizer = BinTokenizer(
            n_bins=self.n_bins,
            bin_type=self.bin_type,
            high=self.n_bins,  # uniform bin size 1
        )

    def __call__(self, embeddings, train=True) -> jax.Array:
        """
        Args:
            embeddings: jnp.ndarray w/ shape (batch_size, horizon, n_tokens, embedding_size)
        """
        if isinstance(embeddings, dict):
            assert (
                self.readout_key is not None
            ), "Must specify readout_key if passing in a dictionary of OrcaTransformer embeddings"
            embeddings = embeddings[self.readout_key]
        batch_size, horizon, n_tokens, embedding_size = embeddings.shape

        embeddings = embeddings.mean(
            axis=-2
        )  # Now, embeddings is (batch_size, horizon, embedding_size)
        logits = self.readout_proj(embeddings)  # (batch_size, horizon, n_bins)
        logits = jnp.reshape(
            logits,
            (
                batch_size,
                horizon,
                self.n_bins,
            ),
        )
        return logits

    def loss(self, embeddings, observations, tasks, pad_mask, train=True):
        """
        Args:
            embeddings: jnp.ndarray w/ shape (batch_size, horizon, num_tokens, embedding_size)
            observations: dict of input observations
            tasks: dict of task information
            pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep.

        Returns:
            loss: float
            metrics: dict
        """

        # (batch, horizon, token_embedding_size)
        distance_logits = self.__call__(embeddings, train=train)
        # compute log probabilities for predicted distances
        distance_logprob = jax.nn.log_softmax(distance_logits, axis=-1)

        # compute target distances
        # use goal distance when it is not padded, otherwise use distance to end (for language instruction tasks)
        distance_to_goal = jnp.maximum(
            tasks["goal_timestep"][:, None] - observations["timestep"], 0
        )
        distance_to_end = jnp.maximum(
            tasks["end_timestep"][:, None] - observations["timestep"], 0
        )
        has_goal = jnp.any(tasks["image_0"], axis=(-1, -2, -3))
        distance_target = jnp.where(
            has_goal[:, None], distance_to_goal, distance_to_end
        )

        # tokenize the target actions and convert them to one hot vectors
        distance_labels = self.distance_tokenizer(distance_target)
        distance_labels_one_hot = jax.nn.one_hot(distance_labels, self.n_bins)

        # compute the CE loss using the log probabilities and target distances
        distance_loss = -jnp.sum(distance_logprob * distance_labels_one_hot, axis=-1)
        # mask the loss with the pad mask to avoid supervising padding
        distance_loss = (distance_loss * pad_mask).mean()

        # take the highest probability distances as the predicted distances
        distance_pred = jnp.argmax(distance_logits, axis=-1)

        # compute accuracy between predicted distances and target distances
        accuracy = distance_pred == distance_labels
        # mask the accuracy with the pad mask to remove the contribution of padding
        accuracy = (accuracy * pad_mask).mean()

        # detokenize the predicted distances
        distance_values = self.distance_tokenizer.decode(distance_pred)
        # compute the mean squared error between predicted distances and target distances
        distance_mse = jnp.square(distance_target - distance_values)
        # mask the mse with the pad mask to remove the contribution of padding
        distance_mse = (distance_mse * pad_mask).mean()

        return distance_loss, {
            "loss": distance_loss,
            "mse": distance_mse,
            "accuracy": accuracy,
            "distance_targets": distance_target,
            "distance_pred": distance_pred,
        }

    def predict_reward(
        self,
        embeddings,
        train: bool = True,
        argmax: bool = False,
        sample_shape: tuple = (),
        rng: PRNGKey = None,
        temperature: float = 1.0,
    ) -> jax.Array:
        # get the logits for the last distance in the horizon
        distance_logits = self.__call__(embeddings, train=train) * temperature
        distance_logits = distance_logits[:, -1]

        if argmax:
            distance_tokens = jnp.argmax(distance_logits, axis=-1).astype(jnp.int32)
        else:
            dist = distrax.Categorical(logits=distance_logits / temperature)
            distance_tokens = dist.sample(seed=rng, sample_shape=sample_shape).astype(
                jnp.int32
            )
        return -1 * self.distance_tokenizer.decode(distance_tokens)
