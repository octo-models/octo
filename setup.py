from setuptools import setup

setup(
    name="octo",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
      "jax>=0.4.20",
      "jaxlib>=0.4.20",
      "dlimp @ git+https://github.com/kvablack/dlimp@d08da3852c149548aaa8551186d619d87375df08",
      "tensorflow_datasets <= 4.9.3",  # Required for gsutil to work as intended. See https://github.com/tensorflow/datasets/issues/5203.
    ],
)
