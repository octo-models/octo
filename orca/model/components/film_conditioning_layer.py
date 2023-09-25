# adapted from https://github.com/google-research/robotics_transformer/blob/master/film_efficientnet/film_conditioning_layer.py

import flax.linen as nn
import jax.numpy as jnp

from orca.utils.typing import *


class FilmConditioning(nn.Module):
    @nn.compact
    def __call__(self, conv_filters: jnp.ndarray, conditioning: jnp.ndarray):
        """Applies FiLM conditioning to a convolutional feature map.

        Args:
            conv_filters: A tensor of shape [batch_size, height, width, channels].
            conditioning: A tensor of shape [batch_size, conditioning_size].

        Returns:
            A tensor of shape [batch_size, height, width, channels].
        """
        projected_cond_add = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)
        projected_cond_mult = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)

        projected_cond_add = projected_cond_add[:, None, None, :]
        projected_cond_mult = projected_cond_mult[:, None, None, :]

        return conv_filters * (1 + projected_cond_add) + projected_cond_mult


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (1, 32, 32, 3))
    x = jnp.array(x)

    z = jnp.ones((1, 64))
    film = FilmConditioning()
    params = film.init(key, x, z)
    y = film.apply(params, x, z)

    print(y.shape)
