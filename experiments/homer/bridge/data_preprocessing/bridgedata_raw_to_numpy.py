"""
Converts data from the BridgeData raw format to numpy format.

Consider the following directory structure for the input data:

    bridgedata_raw/
        rss/
            toykitchen2/
                set_table/
                    00/
                        2022-01-01_00-00-00/
                            collection_metadata.json
                            config.json
                            diagnostics.png
                            raw/
                                traj_group0/
                                    traj0/
                                        obs_dict.pkl
                                        policy_out.pkl
                                        agent_data.pkl
                                        images0/
                                            im_0.jpg
                                            im_1.jpg
                                            ...
                                    ...
                                ...
                    01/
                    ...

The --depth parameter controls how much of the data to process at the
--input_path; for example, if --depth=5, then --input_path should be
"bridgedata_raw", and all data will be processed. If --depth=3, then
--input_path should be "bridgedata_raw/rss/toykitchen2", and only data
under "toykitchen2" will be processed.

The same directory structure will be replicated under --output_path.  For
example, in the second case, the output will be written to
"{output_path}/set_table/00/...".

Squashes images to 128x128.

Can write directly to Google Cloud Storage, but not read from it.

Written by Kevin Black (kvablack@berkeley.edu).
"""
import copy
import glob
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse to the dated directory. Looks for"
    "{input_path}/dir_1/dir_2/.../dir_{depth-1}/2022-01-01_00-00-00/...",
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.9, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("im_size", 128, "Image size")


def squash(path):  # squash from 480x640 to im_size
    im = Image.open(path)
    im = im.resize((FLAGS.im_size, FLAGS.im_size), Image.Resampling.LANCZOS)
    out = np.asarray(im).astype(np.uint8)
    return out


def process_images(path):  # processes images at a trajectory level
    names = sorted(
        [x for x in os.listdir(path) if "images" in x and not "depth" in x],
        key=lambda x: int(x.split("images")[1]),
    )
    image_path = [
        os.path.join(path, x)
        for x in os.listdir(path)
        if "images" in x and not "depth" in x
    ]
    image_path = sorted(image_path, key=lambda x: int(x.split("images")[1]))

    images_out = defaultdict(list)

    tlen = len(glob.glob(image_path[0] + "/im_*.jpg"))

    for i, name in enumerate(names):
        for t in range(tlen):
            images_out[name].append(squash(image_path[i] + "/im_{}.jpg".format(t)))

    images_out = dict(images_out)

    obs, next_obs = dict(), dict()

    for n in names:
        obs[n] = images_out[n][:-1]
        next_obs[n] = images_out[n][1:]
    return obs, next_obs


def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"][:-1], x["full_state"][1:]


def process_time(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["time_stamp"][:-1], x["time_stamp"][1:]


def process_actions(path):  # gets actions
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list


# processes each data collection attempt
def process_dc(path, train_ratio=0.9):
    # a mystery left by the greats of the past
    if "lmdb" in path:
        logging.warning(f"Skipping {path} because uhhhh lmdb?")
        return [], [], [], []

    all_dicts_train = list()
    all_dicts_test = list()
    all_rews_train = list()
    all_rews_test = list()

    # Data collected prior to 7-23 has a delay of 1, otherwise a delay of 0
    date_time = datetime.strptime(path.split("/")[-1], "%Y-%m-%d_%H-%M-%S")
    latency_shift = date_time < datetime(2021, 7, 23)

    search_path = os.path.join(path, "raw", "traj_group*", "traj*")
    all_traj = glob.glob(search_path)
    if all_traj == []:
        logging.info(f"no trajs found in {search_path}")
        return [], [], [], []

    random.shuffle(all_traj)

    num_traj = len(all_traj)
    for itraj, tp in tqdm.tqdm(enumerate(all_traj)):
        try:
            out = dict()

            ld = os.listdir(tp)

            assert "obs_dict.pkl" in ld, tp + ":" + str(ld)
            assert "policy_out.pkl" in ld, tp + ":" + str(ld)
            # assert "agent_data.pkl" in ld, tp + ":" + str(ld) # not used

            obs, next_obs = process_images(tp)
            acts = process_actions(tp)
            state, next_state = process_state(tp)
            time_stamp, next_time_stamp = process_time(tp)
            term = [0] * len(acts)
            if "lang.txt" in ld:
                with open(os.path.join(tp, "lang.txt")) as f:
                    lang = list(f)
                    lang = [l.strip() for l in lang if "confidence" not in l]
            else:
                # empty string is a placeholder for data with no language label
                lang = [""]

            out["observations"] = obs
            out["observations"]["state"] = state
            out["observations"]["time_stamp"] = time_stamp
            out["next_observations"] = next_obs
            out["next_observations"]["state"] = next_state
            out["next_observations"]["time_stamp"] = next_time_stamp

            out["observations"] = [
                dict(zip(out["observations"], t))
                for t in zip(*out["observations"].values())
            ]
            out["next_observations"] = [
                dict(zip(out["next_observations"], t))
                for t in zip(*out["next_observations"].values())
            ]

            out["actions"] = acts
            out["terminals"] = term
            out["language"] = lang

            # shift the actions according to camera latency
            if latency_shift:
                out["observations"] = out["observations"][1:]
                out["next_observations"] = out["next_observations"][1:]
                out["actions"] = out["actions"][:-1]
                out["terminals"] = term[:-1]

            labeled_rew = copy.deepcopy(out["terminals"])[:]
            labeled_rew[-2:] = [1, 1]

            traj_len = len(out["observations"])
            assert len(out["next_observations"]) == traj_len
            assert len(out["actions"]) == traj_len
            assert len(out["terminals"]) == traj_len
            assert len(labeled_rew) == traj_len

            if itraj < int(num_traj * train_ratio):
                all_dicts_train.append(out)
                all_rews_train.append(labeled_rew)
            else:
                all_dicts_test.append(out)
                all_rews_test.append(labeled_rew)
        except FileNotFoundError as e:
            logging.error(e)
            continue
        except AssertionError as e:
            logging.error(e)
            continue

    return all_dicts_train, all_dicts_test, all_rews_train, all_rews_test


def make_numpy(path, train_proportion):
    dirname = os.path.abspath(path)
    outpath = os.path.join(
        FLAGS.output_path, *dirname.split(os.sep)[-(max(FLAGS.depth - 1, 1)) :]
    )

    if os.path.exists(outpath):
        if FLAGS.overwrite:
            logging.info(f"Deleting {outpath}")
            tf.io.gfile.rmtree(outpath)
        else:
            logging.info(f"Skipping {outpath}")
            return

    outpath_train = tf.io.gfile.join(outpath, "train")
    outpath_val = tf.io.gfile.join(outpath, "val")
    tf.io.gfile.makedirs(outpath_train)
    tf.io.gfile.makedirs(outpath_val)

    lst_train = []
    lst_val = []
    rew_train_l = []
    rew_val_l = []

    for dated_folder in os.listdir(path):
        curr_train, curr_val, rew_train, rew_val = process_dc(
            os.path.join(path, dated_folder), train_ratio=train_proportion
        )
        lst_train.extend(curr_train)
        lst_val.extend(curr_val)
        rew_train_l.extend(rew_train)
        rew_val_l.extend(rew_val)

    if len(lst_train) == 0 or len(lst_val) == 0:
        return

    with tf.io.gfile.GFile(tf.io.gfile.join(outpath_train, "out.npy"), "wb") as f:
        np.save(f, lst_train)
    with tf.io.gfile.GFile(tf.io.gfile.join(outpath_val, "out.npy"), "wb") as f:
        np.save(f, lst_val)

    # doesn't seem like these are ever used anymore
    # np.save(os.path.join(outpath_train, "out_rew.npy"), rew_train_l)
    # np.save(os.path.join(outpath_val, "out_rew.npy"), rew_val_l)


def main(_):
    assert FLAGS.depth >= 1

    # each path is a directory that contains dated directories
    paths = glob.glob(os.path.join(FLAGS.input_path, *("*" * (FLAGS.depth - 1))))

    worker_fn = partial(make_numpy, train_proportion=FLAGS.train_proportion)

    with Pool(FLAGS.num_workers) as p:
        list(tqdm.tqdm(p.imap(worker_fn, paths), total=len(paths)))


if __name__ == "__main__":
    app.run(main)
