# ORCA

A robot foundation model.

## Training

```
python experiments/main/train.py --config.data_path=<...> --config.save_dir=<...>
```

## Contributing
Experimental things and training/eval scripts should go in `experiments/<your_name>`. To make any changes to files outside of your experiments directory, please open a pull request.

To enable code checks and auto-formatting, please install pre-commit hooks:
```
pre-commit install
```

## Environment
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
