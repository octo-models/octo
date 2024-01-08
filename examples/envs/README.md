# Octo Evaluation Environments

The `step` and `reset` functions of the Gym environment should return observations with the images, depth images, and/or
proprioceptive information that the model expects as input. Specifically, the returned observations should be dictionaries
of the form:
```
obs = {
    "image_primary": ...,
    "image_wrist": ...,
    ...
    "depth_primary": ...,
    "depth_wrist": ...,
    ...
    "proprio": ...,
}
```

Note that the image keys should be `image_{key}` where `key` is one of the `image_obs_keys` specified in the data loading config used to train the model (typically this is `primary` and/or `wrist`).
If a key is not present in the observation dictionary, the model will substitute it with padding.

Check out the example environments in this folder to help you integrate your own environment!
