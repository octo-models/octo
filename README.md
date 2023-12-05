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

To finetune foundational ORCA policies, you can follow the example command below. You can modify hyperparameters like dataset, batch size etc. in [experiments/dibya/finetune_config.py](finetune_config.py).

```
python finetune.py --config=your_finetune_config.py:mode=head_only --config.pretrained_path=...
```

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

## Run Evaluation in Simulation

To run evaluation on a trained model, you can use the following command:
```bash
# requires pybullet and kinpy to be installed
bash scripts/sim_eval.sh
```

This will spawn a `pybullet` environment with a WidowX robot. Since this environment is not used in training, this is purely for testing the pipeline.

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
