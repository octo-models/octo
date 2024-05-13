from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

default_init = nn.initializers.xavier_uniform


@jax.jit
def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def unet_squaredcos_cap_v2(timesteps, s=0.008):
    t = jnp.linspace(0, timesteps, timesteps + 1) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


class SinusoidalPosEmb(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jax.Array):
        half_features = self.features // 2
        emb = jnp.log(10000) / (half_features - 1)
        emb = jnp.exp(jnp.arange(half_features) * -emb)
        emb = x * emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class Downsample1d(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.Conv(self.features, kernel_size=(3,), strides=(2,))(x)


class Upsample1d(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.ConvTranspose(self.features, kernel_size=(4,), strides=(2,))(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    features: int
    kernel_size: int
    n_groups: int

    @nn.compact
    def __call__(self, x: jax.Array):
        x = nn.Conv(
            self.features,
            kernel_size=(self.kernel_size,),
            strides=1,
            padding=self.kernel_size // 2,
        )(x)
        x = nn.GroupNorm(self.n_groups)(x)
        x = mish(x)
        return x


class ConditionalResidualBlock1D(nn.Module):
    features: int
    kernel_size: int = 3
    n_groups: int = 8
    residual_proj: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array):
        residual = x
        x = Conv1dBlock(
            self.features, kernel_size=self.kernel_size, n_groups=self.n_groups
        )(x)

        cond_features = 2 * self.features
        cond = nn.Dense(cond_features, kernel_init=default_init())(mish(cond))
        scale, bias = jnp.split(cond, 2, axis=-1)
        # Scale, bias are (B, D) and x is shape (B, T, D)
        # We need to broadcast over time, so choose axis = -2
        x = x * jnp.expand_dims(scale, axis=-2) + jnp.expand_dims(bias, axis=-2)
        x = Conv1dBlock(
            self.features, kernel_size=self.kernel_size, n_groups=self.n_groups
        )(x)

        if self.residual_proj:
            residual = nn.Conv(self.features, kernel_size=(1,), strides=1, padding=0)(
                residual
            )

        return x + residual


class ConditionalUnet1D(nn.Module):
    down_features: Tuple[int] = (256, 512, 1024)
    mid_layers: int = 2
    kernel_size: int = 3
    n_groups: int = 8
    time_features: int = 256

    @nn.compact
    def __call__(self, obs, action, time, train: bool = False):
        # Embed the timestep
        time = SinusoidalPosEmb(self.time_features)(time)
        time = nn.Dense(4 * self.time_features, kernel_init=default_init())(time)
        time = mish(time)
        time = nn.Dense(self.time_features, kernel_init=default_init())(time)  # (B, D)
        # Define conditioning as time and observation
        cond = jnp.concatenate((obs, time), axis=-1)

        # Project Down
        hidden_reps = []
        for i, features in enumerate(self.down_features):
            # We always project to the dimension on the first residual connection.
            action = ConditionalResidualBlock1D(
                features,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                residual_proj=True,
            )(action, cond)
            action = ConditionalResidualBlock1D(
                features, kernel_size=self.kernel_size, n_groups=self.n_groups
            )(action, cond)
            if i != 0:
                hidden_reps.append(action)
            if i != len(self.down_features) - 1:
                # If we aren't the last step, downsample
                action = Downsample1d(features)(action)

        # Mid Layers
        for _ in range(self.mid_layers):
            action = ConditionalResidualBlock1D(
                self.down_features[-1],
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
            )(action, cond)

        # Project Up
        for features, hidden_rep in reversed(
            list(zip(self.down_features[:-1], hidden_reps, strict=False))
        ):
            action = jnp.concatenate(
                (action, hidden_rep), axis=-1
            )  # concat on feature dim
            # Always project since we are adding in the hidden rep
            action = ConditionalResidualBlock1D(
                features,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                residual_proj=True,
            )(action, cond)
            action = ConditionalResidualBlock1D(
                features, kernel_size=self.kernel_size, n_groups=self.n_groups
            )(action, cond)
            # Upsample
            action = Upsample1d(features)(action)

        # Should be the same as the input shape
        action = Conv1dBlock(
            self.down_features[0], kernel_size=self.kernel_size, n_groups=self.n_groups
        )(action)
        return action
