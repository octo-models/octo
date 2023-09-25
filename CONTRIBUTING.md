# Contributing
We want to make contributing to this project as easy and transparent as possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. Use `pre-commit` to follow the project code style guidelines.
3. Test that a basic training starts with the debug dataset with: ```
python experiments/main/train.py --config experiments/main/configs/train_config.py:ci_debug_dataset  --name debug```

## Issues
We use GitHub issues for general feature discussion, Q&A and tracking public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue or understand the problem.

## License
By contributing to this project, you agree that your contributions will be licensed under the [LICENSE file](LICENSE) in the root directory of this source tree.
