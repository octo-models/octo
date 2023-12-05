import flax
import jax
import jax.numpy as jnp

from orca.utils.typing import Optional, Sequence


@flax.struct.dataclass
class TokenGroup:
    tokens: jax.Array
    mask: jax.Array

    @classmethod
    def create(cls, tokens: jax.Array, mask: jax.Array = None, **kwargs):
        if mask is None:
            mask = jnp.ones(tokens.shape[:-1])
        assert mask.ndim == tokens.ndim - 1
        return cls(tokens, mask, **kwargs)

    @classmethod
    def concatenate(cls, group_list: Sequence["TokenGroup"], axis=-2):
        data = jnp.concatenate([t.tokens for t in group_list], axis=axis)
        mask = jnp.concatenate([t.mask for t in group_list], axis=axis)
        return cls(data, mask)

    @classmethod
    def split(cls, group: "TokenGroup", indices_or_sections, axis=-2):
        data_list = jnp.split(group.tokens, indices_or_sections, axis=axis)
        mask_list = jnp.split(group.mask, indices_or_sections, axis=axis)
        return [cls(data, mask) for data, mask in zip(data_list, mask_list)]
