## Examples

We provide simple example scripts that demonstrate how to inference and finetune OCTO models,
as well as how to use our data loader independently. We provide the following examples:

|                                                                      |                                                                                                                 |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [OCTO Inference](01_inference_pretrained.ipynb)             | Minimal example for loading and inferencing a pre-trained OCTO model                                            |
| [OCTO Finetuning](02_finetune_new_observation_action.py)    | Minimal example for finetuning a pre-trained OCTO models on a small dataset with new observation + action space |
| [OCTO Rollout](03_eval_finetuned.py)                        | Run a rollout of a pre-trained OCTO policy in a Gym environment                                                 |
| [OCTO Robot Eval](04_eval_finetuned_on_robot.py)            | Evaluate a pre-trained OCTO model on a real WidowX robot                                                        |
| [OpenX Dataloader Intro](05_dataloading.ipynb)              | Walkthrough of the features of our Open X-Embodiment data loader                                                |
| [OpenX PyTorch Dataloader](06_pytorch_oxe_dataloader.py) | Standalone Open X-Embodiment data loader in PyTorch                                                             |
