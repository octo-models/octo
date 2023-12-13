# ORCA Eval Environments

The `step` and `reset` functions of the Gym environment should return observations with the images, depth images, and/or
proprioceptive information that the policy expects as input. Specifically, the returned observations should be dictionaries
of the form:
```
obs = {
    "image_0": ...,
    "image_1": ...,
    ...
    "depth_0": ...,
    "depth_1": ...,
    ...
    "proprio": ...,
}
```

Check out the example environments in this folder to help you integrate your own environment!
