# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer.py
import flax.linen as nn
import jax.numpy as jnp


class TransformerLayer(nn.Module):
    layer_size: int = 4096
    num_heads: int = 8
    feed_forward_size: int = 512
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, attention_mask, train: bool):
        """Calls the layer.

        Args:
        x: Input Tensor of shape `(B, T, dim)`.
        attention_mask: a boolean mask of shape `(B, T, T)`, that prevents
            attention to certain positions. The boolean mask specifies which query
            elements can attend to which key elements, 1 indicates attention and 0
            indicates no attention. Broadcasting can happen for the missing batch
            dimensions and the head dimension.
        train: Python boolean indicating whether the layer should behave in
            train mode (adding dropout) or in inference mode (no dropout).

        Returns:
        y: Output Tensor of shape `(B, T, dim)`. Also return the attention scores
        of shape `(B, T, dim)` or None.
        """
        x1 = nn.LayerNorm(epsilon=1e-6)(x)
        x1 = nn.MultiHeadDotProductAttention(
            qkv_features=self.layer_size,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(x1, x1, attention_mask, deterministic=not train)
        x = x + x1
        y = nn.LayerNorm(epsilon=1e-6)(x)
        ff_y = nn.Dense(features=self.feed_forward_size)(y)
        ff_y = nn.Dropout(rate=self.dropout_rate)(ff_y, deterministic=not train)
        x = x + ff_y
        return x


class Transformer(nn.Module):
    num_layers: int = 1
    layer_size: int = 4096
    num_heads: int = 8
    feed_forward_size: int = 512
    dropout_rate: float = 0.1
    vocab_size: int = 256

    @nn.compact
    def __call__(self, x, attention_mask, train: bool):
        """Calls the layer.

        Args:
        x: Input Tensor of shape `(B, T, dim)`.
        train: Python boolean indicating whether the layer should behave in
            train mode (adding dropout) or in inference mode (no dropout).
        attention_mask: a boolean mask of shape `(B, T, T)`, that prevents
            attention to certain positions. The boolean mask specifies which query
            elements can attend to which key elements, 1 indicates attention and 0
            indicates no attention. Broadcasting can happen for the missing batch
            dimensions and the head dimension.

        Returns:
        x: Output Tensor of shape `(B, T, vocab_size)`. If
        `return_attention_scores`, also return attention scores of
        a list of `layer` of elements with shape `(B, T, dim)`.
        """

        seq_len = x.shape[1]
        batch_size = x.shape[0]

        positions = jnp.eye(seq_len)[jnp.newaxis, :, :]
        positions = jnp.repeat(positions, batch_size, axis=0)

        x = nn.Dense(self.feed_forward_size, name="token_emb")(x)
        x = x + nn.Dense(self.feed_forward_size, name="pos_emb")(positions)

        for i in range(self.num_layers):
            x = TransformerLayer(
                self.layer_size,
                self.num_heads,
                self.feed_forward_size,
                self.dropout_rate,
            )(x, attention_mask, train=train)

        x = nn.Dense(self.vocab_size)(x)
        return x
