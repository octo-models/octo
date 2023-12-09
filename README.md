# ORCA

![](https://github.com/rail-berkeley/orca/workflows/run-debug/badge.svg)
![](https://github.com/rail-berkeley/orca/workflows/pre-commit/badge.svg)

This repo contains code for training and finetuning large robot policies.
Currently, ORCA policies are causal transformer models trained on a diverse mix of robot datasets using BC.

![ORCA model](docs/assets/orca_model.jpeg)

We tokenize **task definitions** (like language instructions or goals), **observations** (like RGB-D images and proprioception)
and **actions**. Given the sequence of input tokens, the model is trained to predict the action tokens.

## Installation
```bash
conda create -n orca python=3.10
conda activate orca
pip install -e .
pip install -r requirements.txt
```
For GPU:
```bash
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]==0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

Test the installation by training on the debug dataset:
```bash
python train.py --config tests/debug_config.py --debug
```

## Training

### Data
We use the RLDS data format and provide fast, parallelized data loaders for policy training. To download the datasets
please reach out to [pertsch@berkeley.edu](mailto:pertsch@berkeley.edu) or download datasets directly from the
**"Open X-Embodiment" repo.

### Finetuning ORCA Policies

To finetune foundational ORCA policies, you can follow the example command below. You can modify hyperparameters like dataset, batch size etc. in [finetune_config.py](finetune_config.py).

```
python finetune.py --config=your_finetune_config.py:mode=head_only --config.pretrained_path=...
```
We offer three finetuning modes depending on the parts of the model that are kept frozen: ```head_only```, ```head_mlp_only``` and ```full``` to finetune the full model. Besides, one can specify the task type to finetune with ```image_conditioned```, ```language_conditioned``` or ```multimodal``` for both. For example, to finetune the full transformer with multimodal inputs use:
```--config=your_finetune_config.py:mode=full,multimodal```

In order to finetune the model to new observation or action spaces, one needs to define this in the config file. We provide an example for finetuning for joint control on the Aloha setup here.

### Base Policy Training

To train foundational ORCA policies, you can follow the example command below. You can modify hyperparameters like
dataset, batch size etc. in [config.py](config.py).
```
python train.py --config config.py:vit_s --name=orca --config.dataset_kwargs.oxe_kwargs.data_dir=... --config.dataset_kwargs.oxe_kwargs.data_mix=oxe_magic_soup ...
```


## Code Structure

|  | File                                                    | Description                                                               |
| --- |---------------------------------------------------------|---------------------------------------------------------------------------|
| Hyperparameters | [config.py](config.py)                                  | Defines all hyperparameters for the training run.                         |
| Training Loop | [train.py](train.py)                                    | Main training script.                                                     |
| Datasets | [dataset.py](orca/data/dataset.py)                      | Functions for creating single / interleaved datasets + data augmentation. |
| Encoders | [tokenizers.py](orca/model/components/tokenizers.py)    | Tokenizers that encode image / text inputs into tokens.                   |
| Model + Objective | [orca_policy.py](orca/model/orca_policy.py)             | Sort tokens into sequence, run forward pass, compute loss.                |
| Visualization | [visualization_lib.py](orca/utils/visualization_lib.py) | Utilities for offline qualitative & quantitative eval.                    |
| Sim Evaluation | [sim_eval.sh](orca/scripts/sim_eval.sh) | Script to run model evaluation.                    |

## Evaluation

To evaluate policies on a robot, first wrap your robot controller in a Gym environment. As an example, see
[widowx_env.py](examples/envs/widowx_env.py) which wraps the robot controller used for the
WidowX robot in BridgeData.

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

Then, write a script that creates your Gym environment, loads the pretrained model, and passes both into the
`run_eval_loop` function from [orca/utils/run_eval.py](orca/utils/run_eval.py). As an example, see [eval_widowx.py](examples/eval.py). A command to run this script can be found in [eval.sh](examples/widowx_eval/eval.sh). **VERY IMPORTANT**: make sure to wrap the Gym environment for your robot in the [UnnormalizeActionProprio](orca/utils/gym_wrappers.py) wrapper to unnormalize/normalize the actions and proprio so that they match what the policy was trained on.

## Contributing
Experimental things and training/eval scripts should go in `experiments/<your_name>`. To make any changes to files outside of your experiments directory, please open a pull request.

Steps to contribute:
1. Fork the repo and create your branch from `master`.
2. Use `pre-commit` to enable code checks and auto-formatting.
3. Test that a basic training starts with the debug dataset with: ```
python train.py --config tests/debug_config.py
```


## FAQ

- **Jax complains about wrong CUDA / CuDNN version**: [Jax picks up on the system CuDNN first](https://github.com/google/jax/issues/17497)
(before using the bundled CUDA), so if you encounter version issues please update your system CUDA / CuDNN
or remove it so Jax uses the bundled packages
