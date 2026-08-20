import numpy as np
import pytest
from matplotlib.figure import Figure

from liana.plotting import lric_lineplot


@pytest.fixture
def curves():
    """Five random LRIC-like curves over a shared radius grid."""
    rng = np.random.default_rng(seed=0)
    return {f"pair_{i}": rng.random(20) + 0.5 for i in range(5)}


@pytest.fixture
def radii():
    return np.linspace(0, 500, 20)


def test_lric_lineplot_small_multiples(radii, curves):
    # small-multiples: returns figure, unused axes hidden
    fig = lric_lineplot(radii, curves, return_fig=True)
    assert isinstance(fig, Figure)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 5


def test_lric_lineplot_overlay(radii, curves):
    fig = lric_lineplot(radii, curves, overlay=True, title="test", return_fig=True)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_lric_lineplot_per_curve_radii(radii):
    # per-curve radii as (r, g) tuples
    mixed = {"a": (radii * 0.5, np.ones(20)), "b": np.ones(20) * 1.2}
    assert isinstance(lric_lineplot(radii, mixed, return_fig=True), Figure)

    # radii=None, all curves carry their own
    all_tuples = {"a": (radii * 0.5, np.ones(20)), "b": (radii * 0.8, np.ones(20) * 1.5)}
    assert isinstance(lric_lineplot(None, all_tuples, overlay=True, return_fig=True), Figure)


def test_lric_lineplot_colors(radii, curves):
    lric_lineplot(radii, curves, colors="red", return_fig=True)
    lric_lineplot(radii, curves, colors=["red", "blue"], return_fig=True)
    lric_lineplot(radii, curves, colors=dict.fromkeys(curves, "green"), return_fig=True)


def test_lric_lineplot_empty_raises(radii):
    with pytest.raises(ValueError, match="at least one entry"):
        lric_lineplot(radii, {}, return_fig=True)


def test_lric_lineplot_into_given_ax(radii, curves):
    from matplotlib import pyplot as plt

    _, ax = plt.subplots()
    fig = lric_lineplot(radii, curves, overlay=True, ax=ax, return_fig=True)
    assert fig is ax.get_figure()


def test_lric_lineplot_bad_colors_raises(radii, curves):
    with pytest.raises(TypeError, match='Unsupported type for `colors`'):
        lric_lineplot(radii, curves, colors=1.0, return_fig=True)
