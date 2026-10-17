"""Regression tests for the matplotlib visualization callback (issue #166)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from octo.utils.visualization_lib import WandBFigure


def test_wandb_figure_exit_produces_rgb_image_matching_canvas():
    wb_figure = WandBFigure(figsize=(2, 3))
    with wb_figure as fig:
        fig.add_subplot().plot([0, 1], [0, 1])
        expected_shape = wb_figure.canvas.get_width_height()[::-1] + (3,)
    assert wb_figure.image.dtype == np.uint8
    assert wb_figure.image.shape == expected_shape
    # the plotted line must actually show up in the rendered image
    assert wb_figure.image.std() > 0


def test_wandb_figure_rgb_image_at_nondefault_dpi():
    wb_figure = WandBFigure(figsize=(5, 5), dpi=50)
    with wb_figure as fig:
        fig.add_subplot().plot([0, 1], [0, 1])
        expected_shape = wb_figure.canvas.get_width_height()[::-1] + (3,)
    assert wb_figure.image.dtype == np.uint8
    assert wb_figure.image.shape == expected_shape
