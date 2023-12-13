import flax.linen as nn
from transformers import CLIPTextConfig, CLIPVisionConfig, FlaxCLIPModel
from transformers.models.clip.modeling_flax_clip import (
    FlaxCLIPTextTransformer,
    FlaxCLIPVisionTransformer,
)

# TODO: this would need to be integrated if we want to use CLIP visual encoder
# def _clip_image_preprocess(image):
#     # this should be exactly the same as HF's CLIPProcessor
#     image = tf.image.resize(image, (224, 224), method="bicubic")
#     image = image / 255.0
#     image = (image - [0.48145466, 0.4578275, 0.40821073]) / [
#         0.26862954,
#         0.26130258,
#         0.27577711,
#     ]
#     return image


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

    def __call__(self, observations, tasks, train: bool = True):
        b, t, h, w, c = observations["image"].shape
        if self.conditioning_type == "obs_image":
            pixel_values = observations["image"]
            pixel_values = pixel_values.reshape((b * t, h, w, c))
            tokens = self.clip_vision_transformer(
                pixel_values=pixel_values, deterministic=not train
            ).last_hidden_state
            tokens = tokens.reshape((b, t, *tokens.shape[1:]))

        elif self.conditioning_type == "goal_image":
            pixel_values = tasks["image"]
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

    def __call__(self, observations, tasks, train: bool = True):
        return self.clip_text_transformer(
            **tasks["language"], deterministic=not train
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

    find_and_replace(params, "clip_vision_transformer", vision_transformer_params)
    find_and_replace(params, "clip_text_transformer", text_transformer_params)
    return params
