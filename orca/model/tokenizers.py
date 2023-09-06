import functools as ft
from typing import Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from orca.model.clip import CLIPTextTokenizer, CLIPVisionTokenizer
from orca.model.transformer import MlpBlock
from orca.model.vision import encoders

EPS = 1e-6


# adapted from https://github.com/google-research/robotics_transformer/blob/master/tokenizers/token_learner.py
class TokenLearner(nn.Module):
    num_tokens: int
    bottleneck_dim: int = 64
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        if len(inputs.shape) == 4:
            inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[-1])
        x = nn.LayerNorm()(inputs)
        x = MlpBlock(
            mlp_dim=self.bottleneck_dim,
            out_dim=self.num_tokens,
            dropout_rate=self.dropout_rate,
        )(x, train=train)
        x = jnp.transpose(x, (0, 2, 1))  # (batch, num_tokens, h*w)
        x = nn.softmax(x, axis=-1)
        return jnp.einsum("bna,baf->bnf", x, inputs)


# adapted from https://github.com/google-research/robotics_transformer/blob/master/tokenizers/image_tokenizer.py
class ImageTokenizer(nn.Module):
    encoder: str
    encoder_kwargs: dict = None
    use_token_learner: bool = False
    num_tokens: int = 8  # this is not enforced unless use_token_learner is True
    conditioning_type: str = "none"
    image_obs_keys: Tuple[str] = ("image_0",)  # list of image keys to tokenize
    stack_image_obs: bool = (
        True  # if True, stacks image observation inputs channel-wise
    )

    @nn.compact
    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        def assemble_image_obs(obs):
            if len(self.image_obs_keys) > 1:
                assert (
                    self.stack_image_obs
                )  # currently only support stacking for multi-image inputs
            return jnp.concatenate([obs[key] for key in self.image_obs_keys], axis=-1)

        # observations["image"] is (batch, obs_horizon, height, width, channel)
        # tasks["image"] is (batch, height, width, channel)
        image = assemble_image_obs(observations)
        b, t, h, w, c = image.shape
        if self.conditioning_type == "none":
            # late-fusion architecture, image encoder doesn't see task and obs together
            image = jnp.reshape(image, (b * t, h, w, c))
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(image)
            image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))
        elif self.conditioning_type == "goal_image":
            # early-fusion goal-image only architecture, concatenate obs and goal image channel-wise
            image = jnp.concatenate([image[:, -1], assemble_image_obs(tasks)], axis=-1)
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(image)
            image_tokens = jnp.reshape(image_tokens, (b, -1, image_tokens.shape[-1]))
        elif self.conditioning_type == "goal_image_no_obs":
            image = assemble_image_obs(tasks)
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(image)
            image_tokens = jnp.reshape(image_tokens, (b, -1, image_tokens.shape[-1]))
        elif self.conditioning_type == "film_language":
            # encode task and pass into encoder with FiLM
            image = jnp.reshape(image, (b * t, h, w, c))
            lang = tasks["language"]
            lang = lang[:, None, :].repeat(t, axis=1)
            lang = jnp.reshape(lang, (b * t, -1))
            image_tokens = encoders[self.encoder](**self.encoder_kwargs)(
                image, cond_var=lang
            )
            image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))

        if self.use_token_learner:
            image_tokens = jnp.reshape(
                image_tokens, (b * t, -1, image_tokens.shape[-1])
            )
            image_tokens = TokenLearner(num_tokens=self.num_tokens)(
                image_tokens, train=train
            )
        return image_tokens


class LanguageTokenizer(nn.Module):
    encoder: str = None
    encoder_kwargs: dict = None
    num_tokens: int = 1

    @nn.compact
    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        # TODO (andre) will need an actual encoder if we want token-level embeddings

        # add a time dimension to language
        if tasks["language"].ndim == 2:
            tokens = tasks["language"][:, None, :]
        else:
            tokens = tasks["language"]

        return tokens


class ActionTokenizer(nn.Module):
    action_dim: int
    vocab_size: int
    normalization_type: str = "bounds"
    low: float = 0
    high: float = 1

    def setup(self):
        if self.normalization_type == "bounds":
            self.thresholds = jnp.linspace(self.low, self.high, self.vocab_size + 1)
        elif self.normalization_type == "normal":
            self.thresholds = norm.ppf(jnp.linspace(EPS, 1 - EPS, self.vocab_size + 1))
        else:
            raise ValueError

    def __call__(self, actions, mode: str = "tokenize"):
        if mode == "tokenize":
            if self.normalization_type == "bounds":
                actions = jnp.clip(actions, self.low + EPS, self.high - EPS)
            actions = actions[..., None]
            token_one_hot = (actions < self.thresholds[1:]) & (
                actions >= self.thresholds[:-1]
            ).astype(jnp.uint8)
            action_tokens = jnp.argmax(token_one_hot, axis=-1)
            return action_tokens
        elif mode == "detokenize":
            action_tokens = actions
            one_hot = jax.nn.one_hot(action_tokens, self.vocab_size)
            bin_avgs = (self.thresholds[1:] + self.thresholds[:-1]) / 2
            actions = jnp.sum(one_hot * bin_avgs, axis=-1)
            return actions


tokenizers = {
    "obs-tokenizer": ft.partial(
        ImageTokenizer,
        conditioning_type="none",
    ),
    "goal-tokenizer": ft.partial(
        ImageTokenizer,
        conditioning_type="goal_image_no_obs",
    ),
    "goal-obs-tokenizer": ft.partial(
        ImageTokenizer,
        conditioning_type="goal_image",
    ),
    "obs-film-language-tokenizer": ft.partial(
        ImageTokenizer,
        conditioning_type="film_language",
    ),
    "language-tokenizer": LanguageTokenizer,
    "clip-obs-tokenizer": ft.partial(
        CLIPVisionTokenizer,
        conditioning_type="obs_image",
    ),
    "clip-goal-tokenizer": ft.partial(
        CLIPVisionTokenizer,
        conditioning_type="goal_image",
    ),
    "clip-text-tokenizer": CLIPTextTokenizer,
    # TODO (andre) other possible tokenizers:
    # "language-wordpiece-tokenizer": use token-level embeddings
    # "proprio": use proprio from observations
}

if __name__ == "__main__":
    import jax
    import numpy as np

    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    action = np.broadcast_to(action, [2, 2, 7])
    tokenizer = ActionTokenizer(
        action_dim=7, vocab_size=256, normalization_type="normal"
    )
    params = tokenizer.init(jax.random.PRNGKey(0), action)
    action_tokens = tokenizer.apply(params, action)
    detokenized_actions = tokenizer.apply(params, action_tokens, mode="detokenize")

    print(action)
    print(action_tokens)
    print(detokenized_actions)
