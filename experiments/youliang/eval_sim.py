import json
import os

from absl import app, flags
import tensorflow as tf

from octo.sim.widowx_sim_env import WidowXSimEnv
from octo.utils.eval_utils import download_checkpoint_from_gcs
from octo.utils.gym_wrappers import UnnormalizeActionProprio
from octo.model.octo_model import PretrainedModel
from octo.utils.run_eval import run_eval_loop

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_multi_integer("checkpoint_step", None, "Checkpoint step", required=True)
flags.DEFINE_string(
    "checkpoint_cache_dir", "/tmp/", "Where to cache checkpoints downloaded from GCS"
)

if __name__ == "__main__":

    def main(_):
        assert len(FLAGS.checkpoint_weights_path) == len(FLAGS.checkpoint_step)

        models = {}
        for weights_path, step in zip(
            FLAGS.checkpoint_weights_path,
            FLAGS.checkpoint_step,
        ):
            weights_path, step = download_checkpoint_from_gcs(
                weights_path,
                step,
                FLAGS.checkpoint_cache_dir,
            )
            assert tf.io.gfile.exists(weights_path), weights_path
            run_name = weights_path.rpartition("/")[2]
            models[f"{run_name}-{step}"] = PretrainedModel.load_pretrained(
                weights_path, step=int(step)
            )

        metadata_path = os.path.join(
            FLAGS.checkpoint_weights_path[0], "dataset_statistics_bridge_dataset.json"
        )
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        env = WidowXSimEnv(image_size=256)
        env = UnnormalizeActionProprio(env, metadata, normalization_type="normal")

        # this is the function that will be called to initialize the goal
        # condition to get the observation
        def get_goal_condition():
            return env.get_observation()

        # run the evaluation loop
        run_eval_loop(env, models, get_goal_condition, 0.1)

    app.run(main)
