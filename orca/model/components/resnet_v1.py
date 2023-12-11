from functools import partial
from typing import Any, Callable, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from orca.model.components.film_conditioning_layer import FilmConditioning
from orca.model.components.vit_encoders import normalize_images

ModuleDef = Any


class AddSpatialCoordinates(nn.Module):
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        grid = jnp.array(
            np.stack(
                np.meshgrid(*[np.arange(s) / (s - 1) * 2 - 1 for s in x.shape[-3:-1]]),
                axis=-1,
            ),
            dtype=self.dtype,
        ).transpose(1, 0, 2)

        if x.ndim == 4:
            grid = jnp.broadcast_to(grid, [x.shape[0], *grid.shape])

        return jnp.concatenate([x, grid], axis=-1)


class SpatialSoftmax(nn.Module):
    height: int
    width: int
    channel: int
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    temperature: None
    log_heatmap: bool = False

    @nn.compact
    def __call__(self, features):
        if self.temperature == -1:
            from jax.nn import initializers

            temperature = self.param(
                "softmax_temperature", initializers.ones, (1), jnp.float32
            )
        else:
            temperature = 1.0

        # add batch dim if missing
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        assert len(features.shape) == 4
        batch_size, num_featuremaps = features.shape[0], features.shape[3]
        features = features.transpose(0, 3, 1, 2).reshape(
            batch_size, num_featuremaps, self.height * self.width
        )

        softmax_attention = nn.softmax(features / temperature)
        expected_x = jnp.sum(
            self.pos_x * softmax_attention, axis=2, keepdims=True
        ).reshape(batch_size, num_featuremaps)
        expected_y = jnp.sum(
            self.pos_y * softmax_attention, axis=2, keepdims=True
        ).reshape(batch_size, num_featuremaps)
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)

        expected_xy = jnp.reshape(expected_xy, [batch_size, 2 * num_featuremaps])

        if no_batch_dim:
            expected_xy = expected_xy[0]
        return expected_xy


class SpatialLearnedEmbeddings(nn.Module):
    height: int
    width: int
    channel: int
    num_features: int = 5
    kernel_init: Callable = nn.initializers.lecun_normal()
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, features):
        """
        features is B x H x W X C
        """
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.height, self.width, self.channel, self.num_features),
            self.param_dtype,
        )

        # add batch dim if missing
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        batch_size = features.shape[0]
        assert len(features.shape) == 4
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2)
        )
        features = jnp.reshape(features, [batch_size, -1])

        if no_batch_dim:
            features = features[0]

        return features


class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNetEncoder(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    norm: str = "group"
    add_spatial_coordinates: bool = False
    pooling_method: str = "none"
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    use_multiplicative_cond: bool = False
    num_spatial_blocks: int = 8
    use_film: bool = False
    img_norm_type: str = "default"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True, cond_var=None):
        expecting_cond_var = self.use_multiplicative_cond or self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"

        x = normalize_images(observations, self.img_norm_type)
        if self.add_spatial_coordinates:
            x = AddSpatialCoordinates(dtype=self.dtype)(x)

        conv = partial(
            self.conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        if self.norm == "batch":
            raise NotImplementedError
        elif self.norm == "group":
            norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        elif self.norm == "layer":
            norm = partial(
                nn.LayerNorm,
                epsilon=1e-5,
                dtype=self.dtype,
            )
        else:
            raise ValueError("norm not found")

        act = getattr(nn, self.act)

        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)

        x = norm(name="norm_init")(x)
        x = act(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=stride,
                    conv=conv,
                    norm=norm,
                    act=act,
                )(x)
                if self.use_film:
                    assert (
                        cond_var is not None
                    ), "Cond var is None, nothing to condition on"
                    x = FilmConditioning()(x, cond_var)
                if self.use_multiplicative_cond:
                    assert (
                        cond_var is not None
                    ), "Cond var is None, nothing to condition on"
                    cond_out = nn.Dense(
                        x.shape[-1], kernel_init=nn.initializers.xavier_normal()
                    )(cond_var)
                    x_mult = jnp.expand_dims(jnp.expand_dims(cond_out, 1), 1)
                    x = x * x_mult

        if self.pooling_method == "spatial_learned_embeddings":
            height, width, channel = x.shape[-3:]
            x = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            )(x)
        elif self.pooling_method == "spatial_softmax":
            height, width, channel = x.shape[-3:]
            pos_x, pos_y = jnp.meshgrid(
                jnp.linspace(-1.0, 1.0, height), jnp.linspace(-1.0, 1.0, width)
            )
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            x = SpatialSoftmax(
                height, width, channel, pos_x, pos_y, self.softmax_temperature
            )(x)
        elif self.pooling_method == "avg":
            x = jnp.mean(x, axis=(-3, -2))
        elif self.pooling_method == "max":
            x = jnp.max(x, axis=(-3, -2))
        elif self.pooling_method == "none":
            pass
        else:
            raise ValueError("pooling method not found")

        return x


ResNet18 = partial(ResNetEncoder, block_cls=ResNetBlock, stage_sizes=(2, 2, 2, 2))
ResNet34 = partial(ResNetEncoder, block_cls=ResNetBlock, stage_sizes=(3, 4, 6, 3))
ResNet50 = partial(
    ResNetEncoder, block_cls=BottleneckResNetBlock, stage_sizes=(3, 4, 6, 3)
)
