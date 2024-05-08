"""
Contains basic logic for randomly zero-ing out keys in the task specification.
"""

import pickle

from huggingface_hub import hf_hub_download
import tensorflow as tf

from octo.data.utils.data_utils import to_padding


def delete_and_rephrase(
    traj,
    paraphrases_repo: str,
    paraphrases_filename: str,
    rephrase_prob: float,
    keep_image_prob: float,
):
    traj = rephrase_instruction(
        traj, paraphrases_repo, paraphrases_filename, rephrase_prob
    )
    traj = delete_task_conditioning(traj, keep_image_prob)
    return traj


class Rephraser:
    def create_static_hash_table(self, dictionary):
        """Takes a python dictionary with string keys and values and creates a tf static hash table"""
        keys = list(dictionary.keys())
        values = list(dictionary.values())
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys, values, key_dtype=tf.string, value_dtype=tf.string
        )
        hash_table = tf.lookup.StaticHashTable(initializer, default_value="")
        return hash_table

    def __init__(self, paraphrases_repo: str, paraphrases_filename: str):
        if isinstance(paraphrases_repo, str) and isinstance(paraphrases_filename, str):
            with open(
                hf_hub_download(
                    repo_id=paraphrases_repo,
                    filename=paraphrases_filename,
                    repo_type="dataset",
                ),
                "rb",
            ) as file:
                lang_paraphrases = pickle.load(file)
                # Create StaticHashTable
                self.rephrase_lookup = self.create_static_hash_table(lang_paraphrases)


def rephrase_instruction(
    traj: dict, paraphrases_repo: str, paraphrases_filename: str, rephrase_prob: float
) -> dict:
    """Randomly rephrases language instructions with precomputed paraphrases
    Args:
       traj: A dictionary containing trajectory data. Should have a "task" key.
       paraphrases_repo: The name of the HF repo containing the paraphrases file.
       paraphrases_filename: The name of the file containing the paraphrases.
       rephrase_prob: The probability of augmenting the language instruction. The probability of keeping the language
           instruction is 1 - rephrase_prob.
    """
    rephraser = Rephraser(paraphrases_repo, paraphrases_filename)

    if "language_instruction" not in traj["task"]:
        return traj
    original_language = traj["task"]["language_instruction"]
    # check the language key is not empty
    string_is_not_empty = tf.reduce_all(tf.strings.length(original_language) > 0)
    # check dict is not empty
    dict_is_not_empty = bool(rephraser.rephrase_lookup)
    if dict_is_not_empty and string_is_not_empty:
        rephrased_instruction = rephraser.rephrase_lookup.lookup(original_language[0])
        rephrased_instruction = tf.where(
            tf.strings.length(rephrased_instruction) > 0,
            original_language[0] + "." + rephrased_instruction,
            original_language[0],
        )
        split_tensor = tf.strings.split(rephrased_instruction, sep=".")
        num_strings = tf.cast(tf.shape(split_tensor)[0], tf.int32)
        random_index = tf.random.uniform(
            (tf.shape(original_language)[0],),
            minval=0,
            maxval=num_strings,
            dtype=tf.int32,
        )
        sampled_language = tf.gather(split_tensor, random_index)
        rand = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
        sampled_language = tf.where(
            rand < rephrase_prob,
            sampled_language,
            original_language,
        )
        traj["task"]["language_instruction"] = sampled_language
    return traj


def delete_task_conditioning(
    traj: dict,
    keep_image_prob: float,
):
    """
    Randomly drops out either the goal images or the language instruction. Only does something if both of
    these are present.

    Args:
        traj: A dictionary containing trajectory data. Should have a "task" key.
        keep_image_prob: The probability of keeping the goal images. The probability of keeping the language
            instruction is 1 - keep_image_prob.
    """
    if "language_instruction" not in traj["task"]:
        return traj

    image_keys = {
        key
        for key in traj["task"].keys()
        if key.startswith("image_") or key.startswith("depth_")
    }
    if not image_keys:
        return traj

    traj_len = tf.shape(traj["action"])[0]
    should_keep_images = tf.random.uniform([traj_len]) < keep_image_prob
    should_keep_images |= ~traj["task"]["pad_mask_dict"]["language_instruction"]

    for key in image_keys | {"language_instruction"}:
        should_keep = should_keep_images if key in image_keys else ~should_keep_images
        # pad out the key
        traj["task"][key] = tf.where(
            should_keep,
            traj["task"][key],
            to_padding(traj["task"][key]),
        )
        # zero out the pad mask dict for the key
        traj["task"]["pad_mask_dict"][key] = tf.where(
            should_keep,
            traj["task"]["pad_mask_dict"][key],
            tf.zeros_like(traj["task"]["pad_mask_dict"][key]),
        )

    # when no goal images are present, the goal timestep becomes the final timestep
    traj["task"]["timestep"] = tf.where(
        should_keep_images,
        traj["task"]["timestep"],
        traj_len - 1,
    )

    return traj
