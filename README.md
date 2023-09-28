# ORCA

This repo contains code for training and finetuning large robot policies.
Currently, ORCA policies are causal transformer models trained on a diverse mix of robot datasets using BC.

![ORCA model](docs/assets/orca_model.jpeg)

We tokenize **task definitions** (like language instructions or goals), **observations** (like RGB-D images and proprioception)
and **actions**. Given the sequence of input tokens, the model is trained to predict the action tokens.

## Installation
```
conda create -n orca python=3.10
conda activate orca
pip install -e .
pip install -r requirements.txt
```
For GPU:
```
pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]==0.4.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

Test the installation by training on the debug dataset:
```
python train.py --config config.py:ci_debug_dataset  --name debug
```

## Training

### Data
We use the RLDS data format and provide fast, parallelized data loaders for policy training. To download the datasets
please reach out to [pertsch@berkeley.edu](mailto:pertsch@berkeley.edu) or download datasets directly from the
**"Open X-Embodiment" repo.

### Base Policy Training

To train foundational ORCA policies, you can follow the example command below. You can modify hyperparameters like
dataset, batch size etc. in [config.py](config.py).
```
python train.py --config config.py:transformer_bc_bridge --name=orca_bridge --config.dataset_kwargs.data_kwargs_list[0].data_dir=<...> --config.save_dir=<...>
```
### TPU Training
1. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
2. Request access to the [RAIL TPU Project](https://console.cloud.google.com/compute/tpus?project=rail-tpus&pli=1)
3. Check free TPUs at http://tpus.kevin.black
4. Login into a TPU with ```gcloud alpha compute tpus tpu-vm ssh [insert_tpu_name] --zone=[insert_tpu_location]```
5. Create personal bucket on the same region to store checkpoints
6. ```python train.py --config config.py:transformer_bc_bridge --name=orca_bridge --config.dataset_kwargs.data_kwargs_list[0].data_dir=gs://your_dataset --config.save_dir=gs://your_bucket```

### TPU Troubleshooting
If you encounter an error of ```TpuStatesManager::GetOrCreate(): no tpu system exists``` reboot and mount nfs:
1. ```sudo reboot now```
2. ```sudo apt -y update && sudo apt install nfs-common```
3. For central1 location ```sudo mkdir -p -m 777 /nfs/nfs1``` and for central2 ```sudo mkdir -p -m 777 /nfs/nfs2```
4. For central1 location ```sudo mount -o rw,intr 10.244.23.202:/nfs1 /nfs/nfs1``` and for central2 ```sudo mount -o rw,intr 10.30.175.26:/nfs2 /nfs/nfs2```

If you get an error ```tensorflow.python.framework.errors_impl.PermissionDeniedError: Error executing an HTTP request: HTTP response code 403 with body``` you might need to authenticate with ```gcloud auth application-default login```

Also, remember that the dataset, bucket and TPU need to be in the same region.

## Code Structure

|  | File                                                    | Description                                                               |
| --- |---------------------------------------------------------|---------------------------------------------------------------------------|
| Hyperparameters | [config.py](config.py)                                  | Defines all hyperparameters for the training run.                         |
| Training Loop | [train.py](train.py)                                    | Main training script.                                                     |
| Datasets | [dataset.py](orca/data/dataset.py)                      | Functions for creating single / interleaved datasets + data augmentation. |
| Encoders | [tokenizers.py](orca/model/components/tokenizers.py)    | Tokenizers that encode image / text inputs into tokens.                   |
| Model + Objective | [orca_policy.py](orca/model/orca_policy.py)             | Sort tokens into sequence, run forward pass, compute loss.                |
| Visualization | [visualization_lib.py](orca/utils/visualization_lib.py) | Utilities for offline qualitative & quantitative eval.                    |

## Contributing
Experimental things and training/eval scripts should go in `experiments/<your_name>`. To make any changes to files outside of your experiments directory, please open a pull request.

Steps to contribute:
1. Fork the repo and create your branch from `master`.
2. Use `pre-commit` to enable code checks and auto-formatting.
3. Test that a basic training starts with the debug dataset with: ```
python experiments/main/train.py --config experiments/main/configs/train_config.py:ci_debug_dataset  --name debug```


## FAQ

- **Jax complains about wrong CUDA / CuDNN version**: [Jax picks up on the system CuDNN first](https://github.com/google/jax/issues/17497)
(before using the bundled CUDA), so if you encounter version issues please update your system CUDA / CuDNN
or remove it so Jax uses the bundled packages
