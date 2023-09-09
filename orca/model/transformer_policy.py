# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.tokenizers import ActionTokenizer
from orca.model.transformer import Transformer
from orca.typing import PRNGKey, Sequence


class TransformerPolicy(nn.Module):
    """
    Transformer that models trajectories.
    """

    observation_tokenizers: Sequence[nn.Module]
    task_tokenizers: Sequence[nn.Module]
    vocab_size: int = 256
    token_embedding_size: int = 512
    num_layers: int = 4
    mlp_dim: int = 1024
    num_heads: int = 8
    dropout_rate: float = 0.1
    horizon: int = 1
    pred_horizon: int = 1
    action_dim: int = 7
    cond_prev_actions: bool = False
    normalization_type: str = "bounds"

    def setup(self):
        self.transformer = Transformer(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )

        self.action_tokenizer = ActionTokenizer(
            action_dim=self.action_dim,
            vocab_size=self.vocab_size,
            normalization_type=self.normalization_type,
        )

        self.tokens_per_action = self.action_dim
        self.tokens_per_obs = sum(tok.num_tokens for tok in self.observation_tokenizers)
        self.tokens_per_task = sum(tok.num_tokens for tok in self.task_tokenizers)
        self.tokens_per_time_step = self.tokens_per_action + self.tokens_per_obs
        self.vocab_proj = nn.Dense(self.vocab_size)
        self.obs_proj = [
            nn.Dense(self.token_embedding_size)
            for _ in range(len(self.observation_tokenizers))
        ]
        self.task_proj = [
            nn.Dense(self.token_embedding_size)
            for _ in range(len(self.task_tokenizers))
        ]
        self.action_embed = nn.Embed(self.vocab_size, self.token_embedding_size)

    def __call__(
        self,
        observations,
        tasks,
        actions,
        train: bool = False,
    ):
        output = self.transformer_call(
            observations,
            tasks,
            actions,
            train=train,
        )

        # get the output for the predicted actions
        action_logits = output[:, -self.pred_horizon :, -self.action_dim :]
        action_logits = self.vocab_proj(action_logits)

        # only use actions in prediction window for supervision
        actions = actions[:, -self.pred_horizon :]

        action_logprob = jax.nn.log_softmax(action_logits, axis=-1)

        action_labels = self.action_tokenizer(actions, mode="tokenize")
        action_labels_one_hot = jax.nn.one_hot(action_labels, self.vocab_size)

        action_loss = -jnp.sum(action_logprob * action_labels_one_hot, axis=-1).mean()

        action_pred = jnp.argmax(action_logits, axis=-1)
        accuracy = (action_pred == action_labels).mean()

        action_values = self.action_tokenizer(action_pred, mode="detokenize")
        action_mse = jnp.square(actions - action_values).sum(axis=-1).mean()

        return {"loss": action_loss, "mse": action_mse, "accuracy": accuracy}

    def transformer_call(
        self,
        observations,
        tasks,
        actions,
        train: bool = False,
    ):
        task_tokens, obs_tokens, action_tokens = self.get_tokens(
            observations, tasks, actions, train=train
        )
        input_tokens = self.assemble_input_tokens(
            task_tokens, obs_tokens, action_tokens
        )
        output = self.transformer(input_tokens, train=train)
        # remove output corresponding to task
        output = output[:, self.tokens_per_task :]

        # unfold time sequence length from token sequence length
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
        actions=None,
        train: bool = False,
        argmax: bool = False,
        rng: PRNGKey = None,
        temperature: float = 1.0,
    ):
        if actions is None:
            assert not self.cond_prev_actions
            actions = jnp.zeros(
                (
                    observations["image_0"].shape[0],
                    self.horizon,
                    self.action_dim,
                )
            )
        else:
            # actions is (batch, horizon-pred_horizon-1, action_dim)
            # because we don't know future actions or the current action.
            # Pad with zeros to full sequence length.
            assert actions.shape[1] == self.horizon - self.pred_horizon - 1
            actions = jnp.concatenate(
                [
                    actions,
                    jnp.zeros(
                        (actions.shape[0], self.pred_horizon + 1, actions.shape[2])
                    ),
                ],
                axis=1,
            )

        output = self.transformer_call(
            observations,
            tasks,
            actions,
            train=train,
        )

        # use the actions in the prediction window
        action_logits = output[:, -self.pred_horizon :, -self.action_dim :]
        action_logits = self.vocab_proj(action_logits)

        if argmax:
            action_tokens = jnp.argmax(action_logits, axis=-1).astype(jnp.int32)
        else:
            action_tokens = jax.random.categorical(
                rng, action_logits / temperature, axis=-1
            ).astype(jnp.int32)
        return self.action_tokenizer(action_tokens, mode="detokenize")

    def get_tokens(self, observations, tasks, actions, train: bool = False):
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

        # (batch, horizon, tokens_per_action, token_embedding_size)
        action_tokens = self.action_tokenizer(actions, mode="tokenize")
        action_tokens = self.action_embed(action_tokens)

        return task_tokens, obs_tokens, action_tokens

    def assemble_input_tokens(self, task_tokens, obs_tokens, action_tokens):
        """
        Mask tokens to be predicted or tokens that should not be conditioned on.
        Combine obs and action tokens.
        Fold time sequence dim into token sequence dim.
        Prepend task tokens.
        """

        # Mask actions if not conditioning on action history
        if not self.cond_prev_actions:
            action_tokens = jnp.zeros(action_tokens.shape)

        # Mask the prediction window
        obs_tokens = self.mask_targets(obs_tokens)
        action_tokens = self.mask_targets(action_tokens)

        # (batch, horizon, tokens_per_time_step, token_embedding_size)
        tokens = jnp.concatenate([obs_tokens, action_tokens], axis=2)

        # (batch, horizon * tokens_per_time_step, token_embedding_size)
        tokens = jnp.reshape(tokens, (tokens.shape[0], -1, tokens.shape[-1]))

        # (batch, horizon * tokens_per_time_step + tokens_per_task, token_embedding_size)
        tokens = jnp.concatenate([task_tokens, tokens], axis=1)
        return tokens

    def mask_targets(self, tokens):
        """
        During training, tokens has shape:
        (batch, horizon, tokens_per_time_step, token_embedding_size)
        and this function masks the tokens in the prediction window.

        During evaluation, tokens has shape:
        (batch, horizon-pred_horizon, tokens_per_time_step, token_embedding_size)
        because we do not have the future tokens. This function adds mask tokens to make a full sequence.
        """
        return jnp.concatenate(
            [
                tokens[:, : self.horizon - self.pred_horizon],
                jnp.zeros((tokens.shape[0], self.pred_horizon, *tokens.shape[2:])),
            ],
            axis=1,
        )
