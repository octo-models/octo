import flax
import jax
import jax.numpy as jnp

from octo.utils.typing import Sequence


@flax.struct.dataclass
class TokenGroup:
    """A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)

    Attributes:
        tokens: jax.Array of shape (..., n_tokens, token_dim)
        mask: jax.Array of shape (..., n_tokens) indicating which tokens are valid (1) vs padding (0)
    """

    tokens: jax.typing.ArrayLike
    mask: jax.typing.ArrayLike

    @classmethod
    def create(
        cls, tokens: jax.typing.ArrayLike, mask: jax.typing.ArrayLike = None, **kwargs
    ):
        if mask is None:
            mask = jnp.ones(tokens.shape[:-1])
        assert mask.ndim == tokens.ndim - 1
        return cls(tokens, mask, **kwargs)

    @classmethod
    def concatenate(cls, group_list: Sequence["TokenGroup"], axis=-2):
        data = jnp.concatenate([t.tokens for t in group_list], axis=axis)
        mask = jnp.concatenate([t.mask for t in group_list], axis=axis + 1)
        return cls(data, mask)
