import functools as ft

from flax.core.frozen_dict import freeze
from transformers import FlaxAutoModel

from orca.model.clip import clip_weights_loader


def hf_weights_loader(hf_model, params):
    model = FlaxAutoModel.from_pretrained(hf_model)
    model_def, model_variables = model.module, model.params
    replaced = False

    def find_and_replace(params, key, replacement):
        nonlocal replaced
        for k in params.keys():
            if k == key:
                params[k] = replacement
                print(f"Replaced {key} in params")
                replaced = True
                return
            if isinstance(params[k], type(params)):
                find_and_replace(params[k], key, replacement)

    params = params.unfreeze()
    find_and_replace(params, "hf_model", model_variables)
    assert replaced, "Failed to load weights"
    return freeze(params)


# index for weight loaders
# these are called to replace parameters after they are initialized from scratch
weights_loaders = {
    "clip": clip_weights_loader,
    "distilbert": ft.partial(hf_weights_loader, "distilbert-base-uncased"),
}
