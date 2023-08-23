from orca.model.clip import clip_weights_loader

# index for weight loaders
# these are called to replace parameters after they are initialized from scratch
weights_loaders = {
    "clip": clip_weights_loader,
}
