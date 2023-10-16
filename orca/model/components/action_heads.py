# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.components import TokenMetadata
from orca.model.components.tokenizers import ActionTokenizer
from orca.utils.typing import PRNGKey


class DiscretizedActionHead(nn.Module):
    """Action head with discretized action prediction and action chunking.

    At each timestep, we have separate action tokens
    for each action dimension and each prediction horizon
    (num_tokens=action_dim*pred_horizon).

    o_tok1, ..., o_tokN, a_tok1, a_token2, ..., a_tok{num_tokens}

    a_tok{k} is tasked with predicting the (k % action_dim)th action dimension, (k // action_dim) timesteps in the future.


    We discretize the action space into a fixed binning, and use
    a categorical loss to predict the bin index for each action dimension.
    Action tokens attend only to task tokens and previous observation tokens.


    """

    horizon: int
    token_embedding_size: int
    window_size: int = 1
    pred_horizon: int = 1
    action_dim: int = 7
    vocab_size: int = 256
    normalization_type: str = "bounds"

    @property
    @nn.nowrap
    def num_tokens(self):
        """Number of transformer tokens used to represent an action in each timestep."""
        return self.action_dim * self.pred_horizon

    @nn.nowrap
    def token_metadata(self, i: int):
        """Any extra metadata needed to determine action mask.

        Args:
            i: int between 0 and self.num_tokens
        Returns: Dict with extra metadata about the token
            (which_pred_horizon, which_action_dim)
        """
        return dict(horizon=i // self.action_dim, action_dim=i % self.action_dim)

    @nn.nowrap
    def attention_mask_ij(
        self, description_i: TokenMetadata, description_j: TokenMetadata
    ):
        """Should token i attend to token j?
        Args:
            description_i: (token_type, token_timestep, extra_metadata)
            description_j: (token_type, token_timestep, extra_metadata)
        Returns: 0 or 1
        """
        assert description_i.name == "action"
        if description_j.name == "task":
            return 1  # Attend to all task tokens
        elif description_j.name == "obs":
            # Attend to all timesteps at same or earlier timestep
            return 1 if description_j.timestep <= description_i.timestep else 0
        elif description_j.name == "action":
            # Don't attend to other action tokens
            return 0
        return 0

    def setup(self):
        assert (
            self.window_size >= self.pred_horizon
        ), "Trajectory must contain enough actions to predict a full chunk."

        self.vocab_proj = nn.Dense(self.vocab_size)
        self.action_tokenizer = ActionTokenizer(
            vocab_size=self.vocab_size,
            normalization_type=self.normalization_type,
        )

    def loss(self, action_embedding, actions, pad_mask):
        """
        Args:
            action_embedding: jnp.ndarray w/ shape (batch_size, horizon, self.num_tokens, self.token_embedding_size)
            actions: jnp.ndarray w/ shape (batch_size, window_size, action_dim)
        Returns:
            loss: float
            metrics: dict
        """

        # get the logits for all the actions by taking the action tokens of each timestep,
        # unfolding the pred_horizon dim, and projecting to the vocab size

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

        # compute log probabilities for predicted actions
        action_logprob = jax.nn.log_softmax(action_logits, axis=-1)

        # chunk the target actions to match the predicted actions
        actions_chunked = self._chunk_actions(actions)

        # only use first horizon timesteps from the window
        actions_chunked = actions_chunked[:, : self.horizon]

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
        action_embedding,
        train: bool = False,
        argmax: bool = False,
        sample_shape: tuple = (),
        rng: PRNGKey = None,
        temperature: float = 1.0,
    ):
        # get the logits for the last action by taking the action tokens of the last timestep,
        # unfolding the pred_horizon dim, and projecting to the vocab size
        # (batch, tokens_per_action, token_embedding_size)
        action_embedding = action_embedding[:, -1, :, :]
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

    def _chunk_actions(self, actions):
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
