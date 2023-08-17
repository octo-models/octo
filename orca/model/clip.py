import flax.linen as nn
import jax
from flax.core.frozen_dict import freeze
from transformers import CLIPTextConfig, CLIPVisionConfig, FlaxCLIPModel
from transformers.models.clip.modeling_flax_clip import (
    FlaxCLIPTextTransformer,
    FlaxCLIPVisionTransformer,
)


class CLIPVisionTokenizer(nn.Module):
    source: str = "openai/clip-vit-base-patch32"
    conditioning_type: str = "obs_image"
    output_dim: int = 512
    num_tokens: int = 50

    def setup(self):
        config = CLIPVisionConfig.from_pretrained(self.source)
        self.clip_vision_transformer = FlaxCLIPVisionTransformer(config)
        # CLIP ViT uses 768 dim tokens but we use 512
        self.projection = nn.Dense(self.output_dim)

    def __call__(self, observations, goals, train: bool = True):
        b, t, h, w, c = observations["image"].shape
        if self.conditioning_type == "obs_image":
            pixel_values = observations["image"]
            pixel_values = pixel_values.reshape((b * t, h, w, c))
            tokens = self.clip_vision_transformer(
                pixel_values=pixel_values, deterministic=not train
            ).last_hidden_state
            tokens = tokens.reshape((b, t, *tokens.shape[1:]))

        elif self.conditioning_type == "goal_image":
            pixel_values = goals["image"]
            tokens = self.clip_vision_transformer(
                pixel_values=pixel_values, deterministic=not train
            ).last_hidden_state
            tokens = tokens.reshape((b, *tokens.shape[1:]))

        else:
            raise NotImplementedError

        tokens = self.projection(tokens)
        return tokens


class CLIPTextTokenizer(nn.Module):
    source: str = "openai/clip-vit-base-patch32"
    # all strings padded to 64 tokens
    num_tokens: int = 64

    def setup(self):
        config = CLIPTextConfig.from_pretrained(self.source)
        self.clip_text_transformer = FlaxCLIPTextTransformer(config)

    def __call__(self, observations, goals, train: bool = True):
        return self.clip_text_transformer(
            **goals["language"], deterministic=not train
        ).last_hidden_state


# after model intialization, call this to load CLIP weights into params
def clip_weights_loader(params):
    clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_def, clip_variables = clip.module, clip.params

    vision_transformer_params = clip_variables["vision_model"]
    text_transformer_params = clip_variables["text_model"]

    def find_and_replace(params, key, replacement):
        for k in params.keys():
            if k == key:
                params[k] = replacement
                print(f"Replaced {key} in params")
                return
            if isinstance(params[k], type(params)):
                find_and_replace(params[k], key, replacement)

    params = params.unfreeze()
    find_and_replace(params, "clip_vision_transformer", vision_transformer_params)
    find_and_replace(params, "clip_text_transformer", text_transformer_params)
    return freeze(params)


if __name__ == "__main__":
    # test CLIPVisionTokenizer
    import jax
    import numpy as np

    rng = jax.random.PRNGKey(0)

    tokenizer = CLIPVisionTokenizer()
    observations = {"image": np.random.randn(2, 4, 224, 224, 3).astype(np.float32)}
    goals = {"image": np.random.randn(2, 224, 224, 3).astype(np.float32)}
    params = tokenizer.init(rng, observations, goals)
    tokens = tokenizer.apply(params, observations, goals)
    print(tokens.shape)
    # (2, 4, 50, 512)

    # test CLIPTextTokenizer
    tokenizer = CLIPTextTokenizer()
    goals = {
        "language": {
            "input_ids": np.random.randint(0, 100, (2, 4)),
            "attention_mask": np.ones((2, 4)),
            "position_ids": np.arange(4),
        }
    }
    params = tokenizer.init(rng, observations, goals)
    tokens = tokenizer.apply(params, observations, goals)
    print(tokens.shape)
    # (2, 4, 512)
