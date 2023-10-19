import flax.linen as nn
import jax.numpy as jnp


class ComputationPlaceholder(nn.Module):
    n_tokens: int
    token_embedding_size: int
    max_horizon: int

    @nn.compact
    def __call__(self, batch_size, horizon):
        posemb_init = nn.initializers.normal(stddev=0.02)  # from BERT.
        embedding = self.param(
            "pos_embedding",
            posemb_init,
            (1, self.max_horizon, self.n_tokens, self.token_embedding_size),
        )
        embedding = embedding[:, :horizon, :, :]
        embedding = jnp.broadcast_to(
            embedding,
            (batch_size, horizon, self.n_tokens, self.token_embedding_size),
        )
        return embedding
