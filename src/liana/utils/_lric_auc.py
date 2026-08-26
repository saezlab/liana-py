from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from liana._docs import d


@d.dedent
def get_lric_auc(
    adata: AnnData,
    uns_key: str,
    max_dist: float | None = None,
    normalize: bool = True,
    min_bins: int = 3,
) -> pd.DataFrame:
    """
    Summarise a ``lric`` / ``cross_pcf`` result into one score per interaction.

    Each interaction's ``g(r)`` profile is reduced to the (signed) area under its
    ``log2 g(r)`` curve: ``> 0`` co-enriched, ``< 0`` depleted, ``0`` random.

    Parameters
    ----------
    %(adata)s
    uns_key
        Key in ``adata.uns`` holding a ``lric`` or ``cross_pcf`` result (the
        ``key_added`` used when it was computed).
    max_dist
        Integrate only over radii ``r < max_dist``; ``None`` uses all radii.
    normalize
        If ``True``, divide the integral by the radial span -- a mean ``log2``
        fold change, comparable across samples. If ``False``, the raw integral.
    min_bins
        Drop interactions with fewer than this many finite ``log2 g(r)`` bins in
        the window (a support gate for ``expr_prop``-masked / degenerate pairs).

    Returns
    -------
    A ``pandas.DataFrame`` with columns ``feature``, ``label`` and ``score``,
    sorted by ``score`` (most enriched first). Empty if nothing clears
    ``min_bins``.

    Examples
    --------
    Rank the interactions of a spatial result by how co-enriched they are across
    radius -- here the cell-type-agnostic LRIC of :func:`liana.mt.lric`:

    >>> import liana as li
    >>> adata = li.testing.generate_toy_spatial()
    >>> li.mt.lric(adata, resource_name='consensus', key_added='lric')
    >>> scores = li.ut.get_lric_auc(adata, uns_key='lric')

    `scores` has `feature`, `label` and `score` columns, sorted most-enriched
    first; a `feature` value passes straight to :func:`liana.pl.lric_lineplot`.
    """
    res = adata.uns[uns_key]
    radii = np.asarray(res["radii"], float)
    win = radii < max_dist if max_dist is not None else np.ones(radii.size, bool)

    # (feature, label, g(r)) per interaction, read from whichever layout was stored
    if "lric" in res:  # agnostic LRIC
        M = np.asarray(res["lric"], float)  # (n_bins, n_pairs)
        items = [(lr, lr.replace("^", " → "), M[:, j]) for j, lr in enumerate(res["pair_names"])]
    elif "pair_names" in res:  # pairwise LRIC
        items = []
        for (sender, receiver), a in res["results"].items():
            A = np.asarray(a, float)  # (n_bins, n_pairs)
            items += [
                (((sender, receiver), lr), f"{sender} → {receiver} | {lr.replace('^', ' → ')}", A[:, j])
                for j, lr in enumerate(res["pair_names"])
            ]
    else:  # cross_pcf (symmetric -- keep each unordered pair once)
        items, seen = [], set()
        for (a, b), c in res["results"].items():
            if frozenset((a, b)) in seen:
                continue
            seen.add(frozenset((a, b)))
            items.append(((a, b), f"{a} ↔ {b}", np.asarray(c, float).ravel()))

    rows = []
    for feature, label, curve in items:
        y = np.log2(np.maximum(np.asarray(curve, float), 0.05))  # floor keeps depleted bins finite
        m = win & np.isfinite(y)
        if m.sum() < min_bins:
            continue
        x, yv = radii[m], y[m]
        span = x.max() - x.min()
        area = np.trapezoid(yv, x)
        rows.append({"feature": feature, "label": label, "score": area / span if (normalize and span > 0) else area})

    cols = ["feature", "label", "score"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("score", ascending=False, ignore_index=True)
