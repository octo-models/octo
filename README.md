# OCTO

![](https://github.com/rail-berkeley/octo/workflows/run-debug/badge.svg)
![](https://github.com/rail-berkeley/octo/workflows/pre-commit/badge.svg)

This repo contains code for training and finetuning OCTO generalist robotic models (GRMs).
OCTO models are transformer-based diffusion policies, trained on a diverse mix of >1M robot trajectories.

![OCTO model](docs/assets/teaser.png)

Out of the box, OCTO supports multiple RGB camera inputs, can control various robot arms,
and can be instructed via language commands or goal images.
OCTO uses a modular attention structure in its transformer backbone, allowing it to be effectively fine-tuned
to robot setups with new sensory inputs, action spaces, and morphologies, using only a small target domain
dataset and accessible compute budgets.


## Installation
```bash
conda create -n octo python=3.10
conda activate octo
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

Test the installation by fine-tuning on the debug dataset:
```bash
python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small
```

## Checkpoints

You can find pre-trained OCTO checkpoints [here](https://huggingface.co/rail-berkeley).
At the moment we provide the following model versions:

| Model                                                         | Inference on 1x NVIDIA 4090 | Size       |
|---------------------------------------------------------------|-----------------------------|------------|
| [OCTO-Base](https://huggingface.co/rail-berkeley/octo-base)   | 13 it/sec                   | 93M Params |
| [OCTO-Small](https://huggingface.co/rail-berkeley/octo-small) | 17 it/sec                   | 27M Params |


## Examples

We provide simple [example scripts](examples) that demonstrate how to inference and finetune OCTO models,
as well as how to use our data loader independently. We provide the following examples:

|                                                                   |                                                                                                                 |
|-------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [OCTO Inference](examples/01_inference_pretrained.ipynb)          | Minimal example for loading and inferencing a pre-trained OCTO model                                            |
| [OCTO Finetuning](examples/02_finetune_new_observation_action.py) | Minimal example for finetuning a pre-trained OCTO models on a small dataset with new observation + action space |
| [OCTO Rollout](examples/03_eval_finetuned.py)                     | Run a rollout of a pre-trained OCTO policy in a Gym environment                                                 |
| [OCTO Robot Eval](examples/04_eval_finetuned_on_robot.py)         | Evaluate a pre-trained OCTO model on a real WidowX robot                                                        |
| [OpenX Dataloader Intro](examples/05_dataloading.ipynb)           | Walkthrough of the features of our Open X-Embodiment data loader                                                |


## OCTO Pre-Training

To reproduce our OCTO pre-training on >1M robot trajectories, run:
```
python scripts/train.py --config scripts/configs/config.py:vit_s --name=octo --config.dataset_kwargs.oxe_kwargs.data_dir=... --config.dataset_kwargs.oxe_kwargs.data_mix=oxe_magic_soup ...
```
You can modify hyperparameters like dataset, batch size etc. in [config.py](scripts/configs/config.py).

To download the pre-training dataset from the [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/),
install the [rlds_dataset_mod package](https://github.com/kpertsch/rlds_dataset_mod)
and run the [prepare_open_x.sh script](https://github.com/kpertsch/rlds_dataset_mod/blob/main/prepare_open_x.sh).
The total size of the pre-processed dataset is ~1.2TB.

We run pre-training using a TPUv4-128 pod in 8 hours for the OCTO-S model and in 14 hours for OCTO-B.


## OCTO Finetuning

We provide a [minimal example](examples/02_finetune_new_observation_action.py) for finetuning with new observations and action space.

We also provide a more advanced finetuning script that allows to change hyperparameters via a config and logs finetuning
metrics. To run advanced finetuning, use:
```
python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small
```

We offer three finetuning modes depending on the parts of the model that are kept frozen: ```head_only```, ```head_mlp_only``` and ```full``` to finetune the full model.
Besides, one can specify the task type to finetune with ```image_conditioned```, ```language_conditioned``` or ```multimodal``` for both.
For example, to finetune the full transformer with image inputs only use:
```--config=finetune_config.py:full,image_conditioned```


## OCTO Evaluation

Loading and inferencing a trained OCTO model is as easy as:
```
from octo.model import OCTOModel

model = OCTOModel.load_pretrained("hf://rail-berkeley/octo-small")
task = model.create_tasks(texts=["pick up the spoon"])
action = model.sample_action(observation, task, rng=jax.random.PRNGKey(0))
```

We provide examples for evaluating OCTO [in a simulated Gym environment](examples/03_eval_finetuned.py) as well
as [on a real WidowX robot](examples/04_eval_finetuned_on_robot.py).

To evaluate on your own environment, simply wrap it in a Gym interface and follow the instructions in the
[Eval Env README](examples/envs/README.md).


## Code Structure

|                     | File                                                    | Description                                                                   |
|---------------------|---------------------------------------------------------|-------------------------------------------------------------------------------|
| Hyperparameters     | [config.py](scripts/configs/config.py)                  | Defines all hyperparameters for the training run.                             |
| Training Loop       | [train.py](scripts/train.py)                            | Main training script.                                                         |
| Finetuning Script   | [finetune.py](scripts/finetune.py)                      | Main finetuning script.                                                       |
| Datasets            | [dataset.py](octo/data/dataset.py)                      | Functions for creating single / interleaved datasets + data augmentation.     |
| Tokenizers          | [tokenizers.py](octo/model/components/tokenizers.py)    | Tokenizers that encode image / text inputs into tokens.                       |
| OCTO Model          | [octo_model.py](octo/model/octo_model.py)               | Main entrypoint for interacting with OCTO models, loading, saving, inference. |
| Model Architecture  | [octo_module.py](octo/model/octo_module.py)             | Combines token sequencing, transformer backbone and readout heads.            |
| Visualization       | [visualization_lib.py](octo/utils/visualization_lib.py) | Utilities for offline qualitative & quantitative eval.                        |


## Contributing
Experimental things and training/eval scripts should go in `experiments/<your_name>`. To make any changes to files outside of your experiments directory, please open a pull request.

Steps to contribute:
1. Fork the repo and create your branch from `master`.
2. Use `pre-commit` to enable code checks and auto-formatting.
3. Test that a basic training starts with the debug dataset with: ```
python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small
```
