import itertools

import matplotlib

matplotlib.use("Agg")
from dataclasses import dataclass
import json
from typing import Any, Dict, Optional, Union

import dlimp as dl
import flax
import gym
import jax
import jax.numpy as jnp
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import tqdm
import wandb

from octo.utils.gym_wrappers import (
    HistoryWrapper,
    RHCWrapper,
    TemporalEnsembleWrapper,
    UnnormalizeActionProprio,
)

BASE_METRIC_KEYS = {
    "mse": ("mse", tuple()),  # What is the MSE
    ####
    #
    # XYZ delta metrics
    #
    ####
    # Angle between true and predicted XYZ delta when moving
    "xyz_angle": (
        "xyz_angle",
        ("moving",),
    ),
    # Did we predict the XYZ delta within 0.5 radians when moving
    "xyz_angle_accuracy": (
        "xyz_angle_accuracy",
        ("moving",),
    ),
    # Did we predict the XYZ delta within 0.5 radians and 50% norm when moving
    "xyz_accuracy": (
        "xyz_accuracy",
        ("moving",),
    ),
    ####
    #
    # Gripper metrics
    #
    ####
    # What % of timesteps (near the actual gripper changes) is the predicted gripper correct?
    "gripping_accuracy": ("gripper_correct", ("gripper_changing",)),
    # Gripper prediction accuracy
    # "gripping_accuracy_full": ("gripper_correct", tuple()),
    # What is the relative height (in m) that we try to grip at, compared to the data?
    "grip_height": ("height_to_grip", ("is_first_grip",)),
    # "early_gripped": ("early_gripped", ("is_first_grip",)),
    # What percentage of grips do we attempt early (early = higher than the height gripped at in the data)
    "early_gripped_height_aware": ("early_gripped_height_aware", ("is_first_grip",)),
    # What timestep do we attempt to grip at (relative to the first timestep we should at)
    "grip_timestep_early": ("timestep_to_grip", ("is_first_grip",)),
}


BASE_SUB_CONDITIONS = dict()


def run_policy_on_trajectory(policy_fn, traj, *, text_processor=None):
    """
    Args:
        policy_fn: A function that takes in a batch of observations and tasks and returns $n$ sampled actions
            of shape (batch_size, n_samples, action_dim). (n_samples can be arbitrary). policy_fn should be
            willing to take in arbitrary batch sizes (use `batched_apply` to wrap a jitted function)
        traj: A dictionary of trajectory data. Should contain "observations", "actions", and "language_instruction" keys.
        text_processor: A function that takes in a batch of text and returns a batch of tokens.
    """
    len_traj = len(traj["action"])

    tasks = {}
    tasks.update(
        jax.tree_map(
            lambda arr: np.tile(arr[-1][-1], (len_traj, *([1] * (arr.ndim - 2)))),
            traj["observation"],
        )
    )
    if text_processor:
        tasks["language_instruction"] = text_processor.encode(
            [s.decode("utf-8") for s in traj["task"]["language_instruction"]]
        )
        tasks["pad_mask_dict"]["language_instruction"] = np.array(
            [len(s.decode("utf-8")) > 0 for s in traj["task"]["language_instruction"]]
        )

    actions = policy_fn(traj["observation"], tasks)

    horizon = jax.tree_util.tree_leaves(traj["observation"])[0].shape[1]
    return {
        "n": np.array(len_traj),
        "pred_actions_chunk": actions,
        "pred_actions": actions[:, :, 0],  # only use first predicted action
        "actions": traj["action"][:, horizon - 1, :],
        "proprio": traj["observation"]["proprio"][:, horizon - 1],
    }


@dataclass
class Visualizer:
    dataset: dl.DLataset
    metric_keys: dict = None
    sub_conditions: dict = None
    freeze_trajs: bool = True  # Use the same trajectories every time
    text_processor: object = None

    def __post_init__(self):
        self.action_proprio_stats = self.dataset.dataset_statistics
        cardinality = self.dataset.cardinality()
        if (
            cardinality == tf.data.INFINITE_CARDINALITY
            or cardinality == tf.data.UNKNOWN_CARDINALITY
        ):
            self.cardinality = float("inf")
        else:
            self.cardinality = cardinality.numpy()
        self.visualized_trajs = False
        self._cached_iterators = {}

    def metrics_for_wandb(
        self,
        infos,
        metric_keys=None,
        sub_conditions=None,
    ):
        """Computes aggregate metrics from a list of trajectory info dictionaries.

        Args:
            infos: Returned from `raw_evaluations`
            metric_keys: A dictionary of metrics to measure.
                k: name of metric (for logging)
                v[0]: name of quantity to measure
                v[1]: names of conditions to mask by
            sub_conditions: A dictionary of sub-conditions to measure. e.g. "when_far" or "when_close"
                k: name of sub-condition (for logging)
                v: names of conditions to mask by
        """

        metric_keys = metric_keys or self.metric_keys or BASE_METRIC_KEYS
        sub_conditions = sub_conditions or self.sub_conditions or BASE_SUB_CONDITIONS

        all_info = {
            k: np.concatenate([info[k] for info in infos])
            for k in infos[0]
            if infos[0][k].ndim > 0
        }

        def masked_mean(quantity_key, *mask_keys):
            mask = np.broadcast_to(
                np.product([all_info[k] for k in mask_keys], axis=0),
                all_info[quantity_key].shape,
            )
            return np.sum(all_info[quantity_key] * mask) / np.sum(mask)

        metrics = {}
        for k, (quantity_key, mask_keys) in metric_keys.items():
            metrics[k] = masked_mean(quantity_key, *mask_keys)
            for sub_condition_name, sub_condition in sub_conditions.items():
                metrics[f"{k}_{sub_condition_name}"] = masked_mean(
                    quantity_key, *mask_keys, *sub_condition
                )
        return metrics

    def visualize_for_wandb(
        self,
        policy_fn,
        max_trajs=1,
        add_images=None,
    ):
        """Returns a dictionary of visualizations to log to wandb.
        Args:
            policy_fn: See `raw_evaluations`
            max_trajs: The maximum number of trajectories to visualize.
            add_images: Whether to add images of the trajectory to the visualization.
        """

        iterator = self.get_iterator(self.dataset, max_trajs)
        visualizations = {}

        for n, traj in tqdm.tqdm(enumerate(iterator), total=max_trajs):
            info = run_policy_on_trajectory(
                policy_fn,
                traj,
                text_processor=self.text_processor,
            )
            info = add_unnormalized_info(info, self.action_proprio_stats)
            info = add_manipulation_metrics(info)

            plotly_fig = plot_trajectory_actions(**info)
            visualizations[f"traj_{n}"] = plotly_fig

            # plot qualitative action trajectory per dimension w/ and w/o action chunk
            visualizations[f"traj_{n}_mpl"] = plot_trajectory_overview_mpl(
                traj, act=info["unnorm_pred_actions_chunk"][:, :, :1], **info
            )
            visualizations[f"traj_{n}_mpl_chunk"] = plot_trajectory_overview_mpl(
                traj, act=info["unnorm_pred_actions_chunk"], **info
            )
            if add_images or not self.visualized_trajs:
                for key in filter(lambda key: "image" in key, traj["observation"]):
                    images = traj["observation"][key][:, 0]

                    observation_slice = np.concatenate(
                        images[np.linspace(0, len(images) - 1, 5).astype(int)], 1
                    )
                    visualizations[f"traj_{n}_{key}"] = wandb.Image(observation_slice)
        self.visualized_trajs = True
        return visualizations

    def raw_evaluations(
        self,
        policy_fn,
        max_trajs=int(1e6),
    ):
        """Computes accuracy metrics for trajectories in the dataset.

        Args:
            policy_fn: A function that takes in a batch of observations and goals and returns sampled actions
                of shape (batch_size, n_samples, action_dim). (n_samples can be arbitrary)
            max_trajs: The maximum number of trajectories to evaluate on.
        Returns:
            all_traj_info: A list of dictionaries containing information about each trajectory (pass into `process_for_wandb`)
        """
        iterator = self.get_iterator(self.dataset, max_trajs)

        all_traj_info = []

        for traj in tqdm.tqdm(iterator, total=max_trajs):
            info = run_policy_on_trajectory(
                policy_fn,
                traj,
                text_processor=self.text_processor,
            )
            info = add_unnormalized_info(info, self.action_proprio_stats)
            info = add_manipulation_metrics(info)
            all_traj_info.append(info)
        return all_traj_info

    def get_iterator(self, dataset, n):
        n = min(n, self.cardinality)
        if n not in self._cached_iterators:
            self._cached_iterators[n] = (
                dataset.take(n).repeat().as_numpy_iterator()
                if self.freeze_trajs
                else dataset.repeat().as_numpy_iterator()
            )
        return itertools.islice(self._cached_iterators[n], n)


@dataclass
class RolloutVisualizer:
    """
    Runs policy rollouts on a given simulated environment.

    Args:
        env_name (str): Gym.make environment creation string
        history_length (int): Number of history steps policy gets conditioned on (window_size).
        action_chunk (int): Number of future steps.
        max_episode_length (int): Max number of steps per rollout episode.
        vis_fps (int): FPS of logged rollout video
        video_subsample_rate (int): Subsampling rate for video logging (to reduce video size for high-frequency control)
        norm_statistics (Union[str, dict], optional): Stats for de-normalizing policy actions & proprio.
        use_temporal_averaging (bool): If true, uses temporal averaging of action chunks during rollout.
    """

    env_name: str
    history_length: int
    action_chunk: int
    max_episode_length: int
    vis_fps: int = 10
    video_subsample_rate: int = 1
    norm_statistics: Optional[Union[str, Dict[str, Any]]] = None
    text_processor: object = None
    use_temp_averaging: bool = False

    def __post_init__(self):
        self._env = gym.make(self.env_name)
        self._env = HistoryWrapper(self._env, self.history_length)
        if self.use_temp_averaging:
            self._env = RHCWrapper(self._env, 1)
            self._env = TemporalEnsembleWrapper(self._env, self.action_chunk)
        else:
            self._env = RHCWrapper(self._env, self.action_chunk)
        if self.norm_statistics:
            if isinstance(self.norm_statistics, str):
                with tf.io.gfile.GFile(self.norm_statistics, "r") as f:
                    norm_stats = json.load(f)
            norm_stats = jax.tree_map(
                lambda x: np.array(x),
                norm_stats,
                is_leaf=lambda x: not isinstance(x, dict),
            )
            self._env = UnnormalizeActionProprio(
                self._env, norm_stats, normalization_type="normal"
            )

    def run_rollouts(self, policy_fn, n_rollouts=10, n_vis_rollouts=3):
        def extract_images(obs):
            # obs has [window_size, ...] shape, only use first time step
            return jnp.concatenate([obs[k][0] for k in obs if "image_" in k], axis=-2)

        def listdict2dictlist(LD):
            return {k: [dic[k] for dic in LD] for k in LD[0]}

        rollout_info = {
            "episode_returns": [],
            "episode_metrics": [],
        }
        for rollout_idx in tqdm.tqdm(range(n_rollouts)):
            obs, info = self._env.reset()
            task = self._env.get_task()
            if jax.tree_util.tree_leaves(task)[0].shape[0] != 1:
                task = jax.tree_map(lambda x: x[None], task)
            if "language_instruction" in task:
                if self.text_processor:
                    task["language_instruction"] = self.text_processor.encode(
                        [s.decode("utf-8") for s in task["language_instruction"]]
                    )
                else:
                    task.pop("language_instruction")
            images = [extract_images(obs)]
            episode_return = 0.0
            metrics = []
            while len(images) < self.max_episode_length:
                # policy outputs are shape [batch, n_samples, pred_horizon, act_dim]
                # we remove batch dimension & use first sampled action, ignoring other samples
                actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)[0, 0]
                obs, reward, done, trunc, info = self._env.step(actions)
                images.extend([extract_images(o) for o in info["observations"]])
                episode_return += reward
                if "metrics" in info:
                    metrics.extend(info["metrics"])
                if done or trunc:
                    break

            rollout_info["episode_returns"].append(episode_return)
            if metrics:
                # concatenate all chunks into one dict of lists, then average across episode
                metrics = listdict2dictlist(metrics)
                rollout_info["episode_metrics"].append(
                    jax.tree_map(lambda x: np.mean(x), metrics)
                )
            if hasattr(self._env, "get_episode_metrics"):
                if metrics:
                    rollout_info["episode_metrics"][-1].update(
                        self._env.get_episode_metrics()
                    )
                else:
                    rollout_info["episode_metrics"].append(
                        self._env.get_episode_metrics()
                    )
            if rollout_idx < n_vis_rollouts:
                # save rollout video
                assert (
                    images[0].dtype == np.uint8
                ), f"Expect uint8, got {images[0].dtype}"
                assert (
                    images[0].shape[-1] == 3
                ), f"Expect [height, width, channels] format, got {images[0].shape}"
                rollout_info[f"rollout_{rollout_idx}_vid"] = wandb.Video(
                    np.array(images).transpose(0, 3, 1, 2)[
                        :: self.video_subsample_rate
                    ],
                    fps=self.vis_fps,
                )
        rollout_info["avg_return"] = np.mean(rollout_info["episode_returns"])
        rollout_info["episode_returns"] = wandb.Histogram(
            rollout_info["episode_returns"]
        )
        metrics = listdict2dictlist(rollout_info.pop("episode_metrics"))
        for metric in metrics:
            rollout_info[metric] = wandb.Histogram(metrics[metric])
            rollout_info[f"avg_{metric}"] = np.mean(metrics[metric])
        return rollout_info


def unnormalize(arr, mean, std, **kwargs):
    return arr * np.array(std) + np.array(mean)


def normalize(arr, mean, std, **kwargs):
    return (arr - np.array(mean)) / np.array(std)


def add_unnormalized_info(
    info,
    normalization_stats,
):
    info.update(
        {
            "unnorm_pred_actions": unnormalize(
                info["pred_actions"], **normalization_stats["action"]
            ),
            "unnorm_pred_actions_chunk": unnormalize(
                info["pred_actions_chunk"], **normalization_stats["action"]
            ),
            "unnorm_actions": unnormalize(
                info["actions"], **normalization_stats["action"]
            ),
            "unnorm_proprio": unnormalize(
                info["proprio"], **normalization_stats["proprio"]
            ),
        }
    )
    return info


def add_manipulation_metrics(info):
    """Adds metrics to the info dictionary from `run_policy_on_trajectory`

    Assumes the following structure of the actions:
        actions[:, :3] = xyz
        actions[:, 3:6] = xyz rotation
        actions[:, 6] = gripper

    Also assumes that unnormalized actions correspond to deltas (measured in meters) from the previous timestep.
    Also assumes that the gripper is closed when the gripper value is > 0.5

    (Note: these are all defaults in the Bridge dataset)
    """
    multiple_sample_info = {k: v for k, v in info.items() if v.ndim == 3}
    single_sample_info = {k: v for k, v in info.items() if v.ndim != 3}

    def per_sample_info(multi_info, single_info):
        kwargs = {**multi_info, **single_info}
        return {
            **_gripper_info(**kwargs),
            **_mse_info(**kwargs),
            **_xyz_info(**kwargs),
            **_condition_info(**kwargs),
            **_gripping_early_metrics(**kwargs),
        }

    new_metrics = jax.vmap(per_sample_info, in_axes=(1, None), out_axes=1)(
        multiple_sample_info, single_sample_info
    )
    return flax.core.copy(info, new_metrics)


def plot_trajectory_actions(
    unnorm_pred_actions,
    unnorm_actions,
    unnorm_proprio,
    **kwargs,
):
    """Creates a 3D plotly figure of the trajectory and predicted actions."""
    pred_actions, actions, proprio = unnorm_pred_actions, unnorm_actions, unnorm_proprio

    # TODO: make this less hardcoded
    proprio = np.concatenate(
        [proprio[..., 1:7], proprio[..., -1:]], axis=-1
    )  # extract proprio

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=proprio[:, 0],
            y=proprio[:, 1],
            z=proprio[:, 2],
            marker=dict(
                size=4,
                color=np.arange(len(proprio)),
                colorscale="Viridis",
            ),
            line=dict(color="darkblue", width=2),
        )
    )

    last_plotted = 0
    for i in range(len(actions) - 1):
        visible = np.linalg.norm((proprio[i] - proprio[last_plotted])[:3]) > 0.05
        visible = visible or (i == 0)
        if visible:
            last_plotted = i

        xs = []
        ys = []
        zs = []
        for action in pred_actions[i]:
            ns = proprio[i] + action
            xs.extend((proprio[i, 0], ns[0]))
            ys.extend((proprio[i, 1], ns[1]))
            zs.extend((proprio[i, 2], ns[2]))
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                visible="legendonly" if not visible else True,
                name="timestep {}".format(i),
                marker=dict(size=1, opacity=0),
                line=dict(color="rgba(0, 0, 255, 0.1)"),
            )
        )
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(
                    x=proprio[0, 0],
                    y=proprio[0, 1],
                    z=proprio[0, 2],
                    text="start",
                    xanchor="left",
                    opacity=0.7,
                ),
                dict(x=proprio[-1, 0], y=proprio[-1, 1], z=proprio[-1, 2], text="goal"),
            ]
        )
    )
    return fig


class WandBFigure:
    def __init__(self, save_to=None, **figure_kwargs):
        self.fig = plt.figure(**figure_kwargs)
        self.canvas = FigureCanvas(self.fig)

    def __enter__(self):
        return plt.figure(self.fig.number)

    def __exit__(self, exc_type, exc_value, traceback):
        self.canvas.draw()
        out_image = np.frombuffer(self.canvas.tostring_rgb(), dtype="uint8")
        self.image = out_image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(self.fig)


def plot_trajectory_overview_mpl(
    traj,
    act,
    unnorm_actions,
    unnorm_proprio,
    **info,
):
    n_act_dims = traj["action"].shape[-1]
    grid_size = int(np.ceil(np.sqrt(n_act_dims + 1)))
    wandb_figure = WandBFigure(figsize=(grid_size * 5, grid_size * 5))
    gs = gridspec.GridSpec(grid_size, grid_size)
    with wandb_figure as fig:
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(info["mse"].mean(axis=1))
        ax.set_ylabel("MSE")
        for i in range(n_act_dims):
            ax = fig.add_subplot(gs[(i + 1) // grid_size, (i + 1) % grid_size])
            ax.plot(unnorm_actions[:, i], label="action")
            # plot predicted action chunks, act.shape = [time, n_samples, chunk, act_dim]
            chunk_length = act.shape[2]
            for t in range(act.shape[0]):
                step_idx, chunk_idx = divmod(t, chunk_length)
                unnorm_pred_actions_i = act[
                    int(step_idx * chunk_length), :, chunk_idx, i
                ]
                x = np.full((unnorm_pred_actions_i.shape[0],), t)
                ax.scatter(
                    x.flat[:],
                    unnorm_pred_actions_i.flat[:],
                    color="tab:red",
                    s=4,
                    alpha=0.5,
                )
                if chunk_idx == 0 and (act.shape[0] // chunk_length) <= 20:
                    ax.axvline(t, color="red", linestyle="--", alpha=0.2)
            ax.set_ylabel(f"dim {i}")
        fig.suptitle(traj["task"]["language_instruction"][0].decode("utf-8"))
    return wandb.Image(wandb_figure.image)


#############################################
#
#
#   A list of metrics to compute on the trajectory
#
#
#############################################


def _get_gripper(actions):
    return actions[:, -1]  # Hard-coded


def _get_xyz(actions):
    return actions[:, :3]  # Hard-coded


def _gripper_closed(actions):
    return _get_gripper(actions) > 0.5  # Hard-coded


def _gripper_correct(unnorm_actions, unnorm_pred_actions, **kwargs):
    return jnp.equal(
        _gripper_closed(unnorm_actions), _gripper_closed(unnorm_pred_actions)
    )


def _xyz_angle(unnorm_actions, unnorm_pred_actions, **kwargs):
    def angle_between(v1, v2):
        v1_u = v1 / (1e-6 + jnp.linalg.norm(v1))
        v2_u = v2 / (1e-6 + jnp.linalg.norm(v2))
        return jnp.arccos(jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0))

    return jax.vmap(angle_between)(
        _get_xyz(unnorm_actions), _get_xyz(unnorm_pred_actions)
    )


def _xyz_close(unnorm_actions, unnorm_pred_actions, **kwargs):
    norm1 = jnp.linalg.norm(_get_xyz(unnorm_actions), axis=-1)
    norm2 = jnp.linalg.norm(_get_xyz(unnorm_pred_actions), axis=-1)
    angle = _xyz_angle(
        unnorm_actions=unnorm_actions, unnorm_pred_actions=unnorm_pred_actions
    )
    return jnp.logical_and(
        angle < 0.5,
        (norm1 > 0.5 * norm2) & (norm2 > 0.5 * norm1),
    )


def _mse(actions, pred_actions, dims=None, **kwargs):
    # Note: this is the MSE of the normalized actions (not the unnormalized actions)
    delta = actions - pred_actions
    if dims is not None:
        delta = delta[:, dims]
    return jnp.sum(delta**2, axis=-1)


def _moving(unnorm_actions, axis=None, magnitude=0, **kwargs):
    if axis is None:
        dist = np.linalg.norm(unnorm_actions[:, :3], axis=1)
    else:
        dist = np.abs(unnorm_actions[:, axis])
    return np.greater(dist, magnitude)


def _xyz_info(**kwargs):
    angle = _xyz_angle(**kwargs)
    return {
        "xyz_angle": angle,
        "xyz_angle_accuracy": angle < 0.5,
        "xyz_accuracy": _xyz_close(**kwargs),
    }


def _mse_info(**kwargs):
    return {
        "mse": _mse(**kwargs),
        "mse_xyz": _mse(dims=[0, 1, 2], **kwargs),  # hard-coded
        "mse_gripper": _mse(dims=[6], **kwargs),  # hard-coded
        "mse_xyzrotation": _mse(dims=[3, 4, 5], **kwargs),  # hard-coded
    }


def _gripping_early_metrics(
    unnorm_actions, unnorm_proprio, unnorm_pred_actions, **kwargs
):
    gripper_closed = _gripper_closed(unnorm_actions)
    pred_gripper_closed = _gripper_closed(unnorm_pred_actions)

    unnorm_proprio = unnorm_proprio[:, 1:]  # Remove special dimension
    z_position = unnorm_proprio[:, 2]

    first_grip = jnp.logical_and(
        gripper_closed, jnp.logical_not(jnp.roll(gripper_closed, 1, axis=0))
    )  # Was the gripper closed at the last timestep?

    gripped_i_steps_early = {
        i: jnp.logical_and(
            first_grip,
            jnp.roll(pred_gripper_closed, i, axis=0),  # Predicted a grip i steps early
        )
        for i in range(1, 5)
    }
    early_gripped = sum(gripped_i_steps_early.values()) > 0

    gripped_i_steps_early_height_aware = {
        i: jnp.logical_and(
            gripped_i_steps_early[i],
            jnp.roll(z_position, i, axis=0) - z_position > 0.005,
        )
        for i in range(1, 5)
    }  # also check that the z position increased
    early_gripped_height_aware = sum(gripped_i_steps_early_height_aware.values()) > 0

    height_to_grip = jnp.zeros_like(z_position)
    timestep_to_grip = jnp.zeros_like(z_position)
    for i in range(1, 5):
        new_height_to_grip = jnp.where(
            jnp.roll(pred_gripper_closed, i, axis=0),
            jnp.roll(z_position, i, axis=0) - z_position,
            0,
        )
        height_to_grip = jnp.maximum(height_to_grip, new_height_to_grip)
        timestep_to_grip = jnp.maximum(
            timestep_to_grip,
            jnp.where(
                jnp.roll(pred_gripper_closed, i, axis=0),
                i,
                0,
            ),
        )
    height_to_grip = jnp.where(first_grip, height_to_grip, 0)
    timestep_to_grip = jnp.where(first_grip, timestep_to_grip, 0)

    gripped_within_two_steps = jnp.logical_and(
        first_grip,
        jnp.logical_or(
            pred_gripper_closed,  # Predicted at this timestep
            jnp.roll(
                pred_gripper_closed, -1, axis=0
            ),  # Predicted at the next timestep. Note that the image of the gripper may already be closed, so this might not be a very reliable metric
        ),
    )
    return {
        "is_first_grip": first_grip,
        "height_to_grip": height_to_grip,
        "early_gripped": early_gripped,
        "early_gripped_height_aware": early_gripped_height_aware,
        "timestep_to_grip": timestep_to_grip,
        "gripped_on_time": gripped_within_two_steps,
    }


def _gripper_info(**kwargs):
    gripper_correct = _gripper_correct(**kwargs)

    actions = kwargs.get("unnorm_actions")
    past_actions = jnp.roll(actions, 3, axis=0)
    future_actions = jnp.roll(actions, -3, axis=0)
    gripping = jnp.logical_or(
        jnp.logical_and(
            _gripper_closed(actions), jnp.logical_not(_gripper_closed(past_actions))
        ),  # Gripper was open in the past, but is closed now
        jnp.logical_and(
            _gripper_closed(future_actions), jnp.logical_not(_gripper_closed(actions))
        ),  # Gripper is open now, but will be closed in the future
    )

    releasing = jnp.logical_or(
        jnp.logical_and(
            _gripper_closed(past_actions), jnp.logical_not(_gripper_closed(actions))
        ),  # Gripper was closed in the past, but is open now
        jnp.logical_and(
            _gripper_closed(actions), jnp.logical_not(_gripper_closed(future_actions))
        ),  # Gripper is closed now, but will be open in the future
    )

    gripper_changing = jnp.logical_or(gripping, releasing)
    still = jnp.logical_not(gripper_changing)
    return {
        "gripper_correct": gripper_correct,
        "gripping": gripping,
        "releasing": releasing,
        "still": still,
        "gripper_changing": gripper_changing,
    }


def _condition_info(**kwargs):
    actions, n = kwargs.get("unnorm_actions"), kwargs.get("n")
    distance = n - np.arange(len(actions))
    return {
        "<10_to_end": distance < 10,
        ">20_to_end": distance > 20,
        "moving": _moving(**kwargs, magnitude=0.01),  # Moved at least 1cm (hard-coded)
    }
