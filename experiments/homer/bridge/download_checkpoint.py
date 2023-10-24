from absl import app, flags
import os

FLAGS = flags.FLAGS

flags.DEFINE_string("bucket_path", None, "Path to save dir on GCP bucket")
flags.DEFINE_string("step_number", None, "Step number")
flags.DEFINE_string(
    "save_path", "/mount/harddrive/homer/checkpoints/", "Path to save checkpoint"
)


def main(_):
    checkpoint_path = os.path.join(FLAGS.bucket_path, f"checkpoint_{FLAGS.step_number}")
    norm_path = os.path.join(FLAGS.bucket_path, "action_proprio*")
    config_path = os.path.join(FLAGS.bucket_path, "config.json*")
    run_name = os.path.basename(os.path.normpath(FLAGS.bucket_path))
    save_path = os.path.join(FLAGS.save_path, run_name)
    os.makedirs(save_path, exist_ok=True)

    os.system(f"gsutil cp -r {checkpoint_path} {save_path}/")
    os.system(f"gsutil cp {norm_path} {save_path}/")
    os.system(f"gsutil cp {config_path} {save_path}/")


if __name__ == "__main__":
    app.run(main)
