import matplotlib

matplotlib.use("Agg")
from dataclasses import dataclass

import dlimp as dl
import flax
import jax
import jax.numpy as jnp
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import tensorflow_datasets as tfds
import tqdm
import wandb

BASE_METRIC_KEYS = {
    "mse": ("mse", tuple()),
    "xyz_angle": ("xyz_angle", ("moving",)),
    "xyz_angle_accuracy": ("xyz_angle_accuracy", ("moving",)),
    "gripping_accuracy": ("gripper_correct", ("gripper_changing",)),
}

BASE_SUB_CONDITIONS = {
    "when_far": (">20_to_end",),
    "when_close": ("<10_to_end",),
}


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
            [s.decode("utf-8") for s in traj["tasks"]["language_instruction"]]
        )

    actions = policy_fn(traj["observation"], tasks)
    return {
        "n": np.array(len_traj),
        "pred_actions": actions,
        "actions": traj["action"][:, -1, :],
        "proprio": traj["observation"]["proprio"][:, -1],
    }


@dataclass
class Visualizer:
    dataset: dl.DLataset
    metric_keys: dict = None
    sub_conditions: dict = None
    cache_trajs: bool = True  # Use the same trajectories for metrics every time
    cache_viz_trajectories: bool = True  # Use same trajs for `visualize` every time
    text_processor: object = None

    def __post_init__(self):
        self.action_proprio_stats = self.dataset.action_proprio_metadata
        self.trajs, self.viz_trajs = [], []
        self.visualized_trajs = False

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
            add_images: Whether to add images of the trajectory to the visualization. If None, will add images if `cache_viz_trajectories` is False
        """

        iterator = self.get_maybe_cached_iterator(
            self.dataset,
            max_trajs,
            self.viz_trajs,
            self.cache_viz_trajectories,
        )
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

            mpl_fig = plot_trajectory_overview_mpl(traj, **info)
            visualizations[f"traj_{n}_mpl"] = mpl_fig
            if (
                add_images
                or not self.cache_viz_trajectories
                or not self.visualized_trajs
            ):
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
        iterator = self.get_maybe_cached_iterator(
            self.dataset,
            max_trajs,
            self.trajs,
            self.cache_trajs,
        )

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

    @staticmethod
    def get_maybe_cached_iterator(dataset, n, cached_trajs, use_cache):
        if not use_cache:
            return dataset.take(n).as_numpy_iterator()
        else:
            if len(cached_trajs) < n:
                new_trajs = list(
                    dataset.take(n - len(cached_trajs)).as_numpy_iterator()
                )
                cached_trajs.extend(new_trajs)
            return cached_trajs[:n]


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
        print(self.fig)
        return plt.figure(self.fig.number)

    def __exit__(self, exc_type, exc_value, traceback):
        self.canvas.draw()
        out_image = np.frombuffer(self.canvas.tostring_rgb(), dtype="uint8")
        self.image = out_image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(self.fig)


def plot_trajectory_overview_mpl(
    traj,
    unnorm_pred_actions,
    unnorm_actions,
    unnorm_proprio,
    **info,
):
    wandb_figure = WandBFigure(figsize=(6, 8))
    gs = gridspec.GridSpec(4, 2)
    with wandb_figure as fig:
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(info["mse"].mean(axis=1))
        ax.set_ylabel("MSE")
        for i in range(7):
            ax = fig.add_subplot(gs[(i + 1) // 2, (i + 1) % 2])
            ax.plot(unnorm_actions[:, i], label="action")
            unnorm_pred_actions_i = unnorm_pred_actions[:, :, i]
            x = np.tile(
                np.arange(len(unnorm_pred_actions_i))[:, None],
                (1, unnorm_pred_actions_i.shape[1]),
            )
            ax.scatter(
                x.flat[:],
                unnorm_pred_actions_i.flat[:],
                color="tab:red",
                s=4,
                alpha=0.5,
            )
            ax.set_ylabel(f"dim {i}")
        fig.suptitle(traj["tasks"]["language_instruction"][0].decode("utf-8"))
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
    }


def _mse_info(**kwargs):
    return {
        "mse": _mse(**kwargs),
        "mse_xyz": _mse(dims=[0, 1, 2], **kwargs),  # hard-coded
        "mse_gripper": _mse(dims=[6], **kwargs),  # hard-coded
        "mse_xyzrotation": _mse(dims=[3, 4, 5], **kwargs),  # hard-coded
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
