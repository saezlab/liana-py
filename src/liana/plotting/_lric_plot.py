import numpy as np
import pandas as pd
import plotnine as p9
from anndata import AnnData
from matplotlib.figure import Figure

from liana._constants import DefaultValues as V
from liana._docs import d


@d.dedent
def lric_lineplot(
    adata: AnnData,
    uns_key: str,
    feature,
    max_dist: float | None = None,
    figure_size: tuple[float, float] = (6, 4),
    return_fig: bool = V.return_fig,
) -> Figure:
    """
    Plot the g(r) profile of a single interaction from ``lric`` / ``cross_pcf``.

    ``g(r)`` is the observed count at radius ``r`` relative to the closed-form
    random-labelling null; ``g(r) > 1`` is co-enrichment, ``< 1`` depletion, and
    the dashed line at ``1`` marks the null. For pairwise LRIC the full coupling
    is decomposed into its architecture-only (``g_pcf``) and expression-only
    (``g_expr``) components, drawn as separate lines.

    Parameters
    ----------
    %(adata)s
    uns_key
        Key in ``adata.uns`` holding a ``lric`` or ``cross_pcf`` result (the
        ``key_added`` used when it was computed).
    feature
        The interaction to plot, matching the stored result:
        ``(sender, receiver)`` for ``cross_pcf``; a ``"ligand^receptor"`` string
        for agnostic LRIC; and ``((sender, receiver), "ligand^receptor")`` for
        pairwise LRIC.
    max_dist
        If given, draw a dotted vertical line at this radius to mark the
        short-range window of interest.
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    A ``plotnine.ggplot`` if ``return_fig`` else ``None`` (draws the plot).
    """
    res = adata.uns[uns_key]
    radii = np.asarray(res["radii"], float)

    if "lric" in res:  # agnostic LRIC -- feature = "ligand^receptor"
        j = list(res["pair_names"]).index(feature)
        curves = {"g(r)": res["lric"][:, j]}
        title = feature
    elif "pair_names" in res:  # pairwise LRIC -- feature = ((sender, receiver), "ligand^receptor")
        (sender, receiver), lr = feature
        j = list(res["pair_names"]).index(lr)
        curves = {
            "g (full)": res["results"][sender, receiver][:, j],
            "g_pcf": res["g_pcf"][sender, receiver],  # (n_bins,), shared across LR pairs
            "g_expr": res["g_expr"][sender, receiver][:, j],
        }
        title = f"{sender} -> {receiver}: {lr}"
    else:  # cross_pcf -- feature = (sender, receiver), symmetric
        key = feature if feature in res["results"] else feature[::-1]
        curves = {"g(r)": res["results"][key]}
        title = f"{feature[0]} vs {feature[1]}"

    df = pd.concat(
        [pd.DataFrame({"radius": radii, "g": np.asarray(y, float), "curve": label}) for label, y in curves.items()],
        ignore_index=True,
    )

    p = (
        p9.ggplot(df, p9.aes("radius", "g", color="curve"))
        + p9.geom_hline(yintercept=1, linetype="dashed", color="grey")
        + p9.geom_line()
        + p9.geom_point(size=1.2)
        + p9.labs(x="Radius (r)", y="g(r)", title=title, color="")
        + p9.theme_bw()
        + p9.theme(figure_size=figure_size)
    )
    if len(curves) == 1:  # single curve -> the legend adds nothing
        p = p + p9.theme(legend_position="none")
    if max_dist is not None:
        p = p + p9.geom_vline(xintercept=max_dist, linetype="dotted", color="black")

    if return_fig:
        return p

    p.draw()
