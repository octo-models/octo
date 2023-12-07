"""
This script demonstrates how to load and rollout a finetuned ORCA model.
We use the ORCA model finetuned on ALOHA sim data from the examples/finetune_aloha.py script.
"""
from absl import app, flags, logging
import gym
import jax
import numpy as np
import wandb

# TODO: move AlohaGymEnv into examples and remove dependency on files in "experiments"
from experiments.lucy.aloha_wrapper import AlohaGymEnv
from orca.utils.gym_wrappers import HistoryWrapper, RHCWrapper, UnnormalizeActionProprio
from orca.utils.pretrained_utils import PretrainedModel

FLAGS = flags.FLAGS

flags.DEFINE_string("finetuned_path", "Path to finetuned ORCA checkpoint directory.")


def main(_):
    # setup wandb for logging
    wandb.init(name="eval_aloha", project="orca")

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = PretrainedModel.load_pretrained(FLAGS.finetuned_path)

    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_0": ...
    #     "image_1": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_0": ...
    #       "image_1": ...
    #     }
    #   }
    ##################################################################################################################
    env = gym.make("aloha-sim-cube-v0")

    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=50)

    # wrap env to handle action/proprio normalization -- match normalization type to the one used during finetuning
    norm_stats = PretrainedModel.get_norm_stats(FLAGS.finetuned_path)
    env = UnnormalizeActionProprio(env, norm_stats, normalization_type="normal")

    # jit model action prediction function for faster inference
    policy_fn = jax.jit(model.sample_actions)

    # running rollouts
    for _ in range(10):
        obs, info = env.reset()

        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = env.get_task()["language_instruction"]
        task = model.create_tasks(texts=language_instruction)

        # run rollout for 400 steps
        images = [obs["image_0"]]
        episode_return = 0.0
        while len(images) < 400:
            # model returns actions of shape [batch, horizon, action_dim] -- remove batch and horizon since they are 1
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)[0, 0]

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            images.extend([o["image_0"] for o in info["observations"]])
            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        wandb.log(
            {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
        )


if __name__ == "__main__":
    app.run(main)
