from absl import app, flags
import os

FLAGS = flags.FLAGS

flags.DEFINE_string("run_name", None, "WandB run name")
flags.DEFINE_string("step_number", None, "Step number")
flags.DEFINE_string("bucket", None, "GCP bucket")


def main(_):
    os.makedirs(f"checkpoints/{FLAGS.run_name}", exist_ok=True)
    os.system(
        f"gsutil cp -r gs://{FLAGS.bucket}/log/orca/{FLAGS.run_name}/checkpoint_{FLAGS.step_number} checkpoints/{FLAGS.run_name}/"
    )
    os.system(
        f"gsutil cp gs://{FLAGS.bucket}/log/orca/{FLAGS.run_name}/action_proprio* checkpoints/{FLAGS.run_name}/"
    )
    os.system(
        f"gsutil cp gs://{FLAGS.bucket}/log/orca/{FLAGS.run_name}/config.json checkpoints/{FLAGS.run_name}/"
    )


if __name__ == "__main__":
    app.run(main)
