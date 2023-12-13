from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf

MULTI_MODULE = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"


class TextProcessor(ABC):
    """
    Base class for text tokenization or text embedding.
    """

    @abstractmethod
    def encode(self, strings: Sequence[str]):
        raise NotImplementedError


class HFTokenizer(TextProcessor):
    def __init__(
        self,
        tokenizer_name: str,
        tokenizer_kwargs: Optional[dict] = {
            "max_length": 64,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
        },
        encode_with_model: bool = False,
    ):
        from transformers import AutoTokenizer, FlaxAutoModel  # lazy import

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_kwargs = tokenizer_kwargs
        self.encode_with_model = encode_with_model
        if self.encode_with_model:
            self.model = FlaxAutoModel.from_pretrained(tokenizer_name)

    def encode(self, strings: Sequence[str]):
        # this creates another nested layer with "input_ids", "attention_mask", etc.
        inputs = self.tokenizer(
            strings,
            **self.tokenizer_kwargs,
        )
        if self.encode_with_model:
            return np.array(self.model(**inputs).last_hidden_state)
        else:
            return dict(inputs)


class MuseEmbedding(TextProcessor):
    def __init__(self):
        import tensorflow_hub as hub  # lazy import
        import tensorflow_text  # noqa: F401

        self.muse_model = hub.load(MULTI_MODULE)

    def encode(self, strings: Sequence[str]):
        with tf.device("/cpu:0"):
            return self.muse_model(strings).numpy()


class CLIPTextProcessor(TextProcessor):
    def __init__(
        self,
        tokenizer_kwargs: Optional[dict] = {
            "max_length": 64,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
        },
    ):
        from transformers import CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.kwargs = tokenizer_kwargs

    def encode(self, strings: Sequence[str]):
        inputs = self.processor(
            text=strings,
            **self.kwargs,
        )
        inputs["position_ids"] = np.expand_dims(
            np.arange(inputs["input_ids"].shape[1]), axis=0
        ).repeat(inputs["input_ids"].shape[0], axis=0)
        return inputs
