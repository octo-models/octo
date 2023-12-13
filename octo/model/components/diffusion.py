# copied from: https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/main/jaxrl_m/networks/diffusion_nets.py
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

default_init = nn.initializers.xavier_uniform


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


class ScoreActor(nn.Module):
    time_preprocess: nn.Module
    cond_encoder: nn.Module
    reverse_network: nn.Module

    def __call__(self, obs_enc, actions, time, train=False):
        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff, train=train)
        reverse_input = jnp.concatenate([cond_enc, obs_enc, actions], axis=-1)
        eps_pred = self.reverse_network(reverse_input, train=train)
        return eps_pred


class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jax.Array):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable = nn.swish
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activation(x)
        return x


class MLPResNetBlock(nn.Module):
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike, train: bool = False) -> jax.Array:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activation,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        x = self.activation(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x


def create_diffusion_model(
    out_dim: int,
    time_dim: int,
    num_blocks: int,
    dropout_rate: float,
    hidden_dim: int,
    use_layer_norm: bool,
):
    return ScoreActor(
        FourierFeatures(time_dim, learnable=True),
        MLP((2 * time_dim, time_dim)),
        MLPResNet(
            num_blocks,
            out_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        ),
    )
