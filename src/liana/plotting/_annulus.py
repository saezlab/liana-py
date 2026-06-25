import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.patches import Annulus

from liana._constants import DefaultValues as V
from liana._constants import Keys as K
from liana._docs import d


@d.dedent
def annulus_plot(
    adata: AnnData,
    spatial_key: str = K.spatial_key,
    annulus_width: float = 20.0,
    radius_step: float = 20.0,
    n_rings: int = 10,
    seed: int = V.seed,
    figure_size: tuple = (6, 6),
) -> None:
    """
    Visualise concentric annuli around a randomly chosen cell on a tissue section.

    Useful for inspecting the local neighbourhood structure and choosing sensible
    ``radius_step`` / ``annulus_width`` parameters before running spatial statistics
    (e.g. cross-PCF or LRIC).

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    annulus_width
        Radial width of each annulus ring (in coordinate units, e.g. µm).
        Setting ``annulus_width == radius_step`` avoids double-counting.
    radius_step
        Step size between successive ring inner radii (in the same units as
        the spatial coordinates).
    n_rings
        Number of concentric rings to draw.
    %(seed)s
    %(figure_size)s

    Raises
    ------
    KeyError
        If ``spatial_key`` is not found in ``adata.obsm``.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"'{spatial_key}' not found in adata.obsm.")

    coords = adata.obsm[spatial_key]

    sel_inner = np.arange(1, n_rings + 1, dtype=float) * radius_step
    sel_outer = sel_inner + annulus_width

    rng = np.random.default_rng(seed)
    center = coords[rng.integers(len(coords))]
    dists = np.linalg.norm(coords - center, axis=1)

    counts = [
        int(np.sum((dists >= r_in) & (dists < r_out)))
        for r_in, r_out in zip(sel_inner, sel_outer)
    ]

    ring_colors = plt.cm.plasma(np.linspace(0.05, 0.90, n_rings))
    view_r = sel_outer[-1] * 1.2
    near_mask = dists <= view_r

    fig, ax = plt.subplots(figsize=figure_size)

    ax.scatter(
        coords[near_mask, 0],
        coords[near_mask, 1],
        s=8,
        c="lightgrey",
        alpha=0.6,
        linewidths=0,
        zorder=1,
        rasterized=True,
    )

    for r_in, r_out, color in zip(sel_inner[::-1], sel_outer[::-1], ring_colors[::-1]):
        ax.add_patch(
            Annulus(center, r=r_out, width=r_out - r_in, color=color, alpha=0.22, zorder=2)
        )
        ax.add_patch(
            plt.Circle(center, r_out, fill=False, edgecolor=color, lw=1.8, zorder=4)
        )
        ax.add_patch(
            plt.Circle(center, r_in, fill=False, edgecolor=color, lw=0.8, ls="--", zorder=4)
        )

    for r_in, r_out, color, count in zip(sel_inner, sel_outer, ring_colors, counts):
        mid_r = (r_in + r_out) / 2
        lx = center[0] + mid_r * np.cos(np.pi / 4)
        ly = center[1] + mid_r * np.sin(np.pi / 4)
        ax.text(
            lx,
            ly,
            f"n={count}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.22", facecolor=color, edgecolor="none", alpha=0.92
            ),
        )

    ax.scatter(*center, s=180, c="black", marker="*", zorder=10)

    ax.set_xlim(center[0] - view_r, center[0] + view_r)
    ax.set_ylim(center[1] - view_r, center[1] + view_r)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_title("Annuli around centre cell (★)", fontsize=10, pad=8)

    ring_patches = [
        mpatches.Patch(
            facecolor=c,
            alpha=0.7,
            label=f"Ring {i + 1}: [{r_in:.0f}–{r_out:.0f}]",
        )
        for i, (r_in, r_out, c) in enumerate(zip(sel_inner, sel_outer, ring_colors))
    ]
    ax.legend(
        handles=ring_patches,
        loc="lower right",
        fontsize=6.5,
        frameon=True,
        framealpha=0.9,
        title="Annuli",
        title_fontsize=7,
        ncol=2,
    )

    plt.tight_layout()
    plt.show()
