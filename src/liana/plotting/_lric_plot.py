import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from liana._constants import DefaultValues as V
from liana._docs import d


@d.dedent
def lric_lineplot(
    radii,
    curves: dict,
    *,
    overlay: bool = False,
    ncols: int = 4,
    colors=None,
    title: str | None = None,
    figure_size: tuple | None = None,
    ax=None,
    return_fig: bool = V.return_fig,
) -> Figure | None:
    """
    Line plot for cross-PCF / LRIC g(r) curves with a baseline at 1.

    Parameters
    ----------
    radii
        1-D array of radius values shared by all curves.  Individual curves
        may supply their own radii by passing ``(radii, g)`` tuples as values.
    curves
        ``{label: g}`` or ``{label: (radii, g)}``.  In small-multiples mode
        labels become subplot titles; in overlay mode they become legend entries.
    overlay
        ``False`` (default) — one subplot per curve arranged in a grid.
        ``True`` — all curves drawn on a single axis.
    ncols
        Number of columns for the small-multiples grid (ignored when
        ``overlay=True``).
    colors
        Colour specification.  Accepts a single colour string (applied to all
        curves), a list (one per curve, in dict-insertion order), or a dict
        keyed by label.  Defaults to the matplotlib tab10 cycle.
    title
        Figure-level super-title.
    %(figure_size)s
    ax
        Existing :class:`~matplotlib.axes.Axes` to draw into.  Only used when
        ``overlay=True``; ignored otherwise.
    %(return_fig)s

    Returns
    -------
    :class:`~matplotlib.figure.Figure` if ``return_fig`` is ``True``,
    otherwise ``None``.
    """
    labels = list(curves.keys())
    n = len(labels)

    if n == 0:
        raise ValueError("`curves` must contain at least one entry.")

    color_map = _resolve_colors(labels, colors)

    if overlay:
        fig, ax = _ensure_ax(ax, figure_size or (6, 4))
        for label in labels:
            r, g = _unpack(radii, curves[label])
            ax.plot(r, g, lw=2, marker="o", ms=4,
                    color=color_map[label], label=label)
        ax.axhline(1, linestyle=":", color="0.3", lw=1.4, label="g(r) = 1")
        ax.set_xlabel("Radius (µm)")
        ax.set_ylabel("g(r)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        if title:
            ax.set_title(title, fontsize=10)
        _finish(fig, title=None, return_fig=return_fig)
        return fig if return_fig else None

    # small multiples
    nrows = math.ceil(n / ncols)
    default_w = min(ncols, n) * 3
    default_h = nrows * 3
    fig, axes = plt.subplots(
        nrows, min(ncols, n),
        figsize=figure_size or (default_w, default_h),
        layout="constrained",
        squeeze=False,
    )
    all_axes = list(axes.flat)
    for ax_i, label in zip(all_axes, labels):
        r, g = _unpack(radii, curves[label])
        ax_i.plot(r, g, lw=2, marker="o", ms=4, color=color_map[label])
        ax_i.axhline(1, linestyle=":", color="0.3", lw=1.2)
        ax_i.set_title(label, fontsize=9)
        ax_i.set_xlabel("Radius (µm)", fontsize=8)
        ax_i.set_ylabel("g(r)", fontsize=8)
        ax_i.grid(alpha=0.2)
        ax_i.tick_params(labelsize=7)

    for ax_i in all_axes[n:]:
        ax_i.set_visible(False)

    _finish(fig, title=title, return_fig=return_fig)
    return fig if return_fig else None


# ── helpers ──────────────────────────────────────────────────────────────────


def _unpack(shared_radii, value):
    if isinstance(value, tuple) and len(value) == 2:
        return np.asarray(value[0], dtype=float), np.asarray(value[1], dtype=float)
    return np.asarray(shared_radii, dtype=float), np.asarray(value, dtype=float)


def _resolve_colors(labels, colors):
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return {lbl: cycle[i % len(cycle)] for i, lbl in enumerate(labels)}
    if isinstance(colors, str):
        return {lbl: colors for lbl in labels}
    if isinstance(colors, list):
        return {lbl: colors[i % len(colors)] for i, lbl in enumerate(labels)}
    if isinstance(colors, dict):
        return colors
    raise TypeError(f"Unsupported type for `colors`: {type(colors)}")


def _ensure_ax(ax, figure_size):
    if ax is not None:
        return ax.get_figure(), ax
    fig, ax = plt.subplots(figsize=figure_size)
    return fig, ax


def _finish(fig, title, return_fig):
    if title:
        fig.suptitle(title, fontsize=12)
    if not return_fig:
        plt.show()
