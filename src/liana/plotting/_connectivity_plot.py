from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from plotnine import aes, geom_point, ggplot, labs, theme, theme_minimal

from liana._core._common import _logg
from liana._core._constants import DefaultValues as V
from liana._core._constants import Keys as K
from liana._core._docs import d
from liana._core._types import get_coordinates


@d.dedent
def connectivity(
    adata: AnnData,
    idx: int,
    spatial_key: str = K.spatial_key,
    connectivity_key: str = K.connectivity_key,
    size: float = 1,
    figure_size: tuple[float, float] = (5.4, 5),
    return_fig: bool = V.return_fig,
) -> ggplot | None:
    """
    Plot spatial connectivity weights.

    Parameters
    ----------
    %(adata)s
    idx
        Column index of the connectivity weights to plot.
    %(spatial_key)s
    %(connectivity_key)s
    size
        Size of the points
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    The resulting connectivity plot.

    Raises
    ------
    AssertionError
        If `connectivity_key` or `spatial_key` are not in `adata.obsp` or `adata.obsm` respectively.

    Examples
    --------
    `idx` picks one spot, and the plot shows how strongly every other spot is
    connected to it under the kernel of :func:`liana.pp.spatial_neighbors`:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> p = li.pl.connectivity(adata, idx=0)
    """
    assert connectivity_key in list(adata.obsp.keys())
    assert spatial_key in adata.obsm_keys()

    _logg(
        "This function will be deprecated in the next version. "
        + "Please use scanpy or squidpy for plotting spatial connectivities.",
        level="warn",
    )

    coordinates = pd.DataFrame(get_coordinates(adata, spatial_key), index=adata.obs_names, columns=["x", "y"]).copy()
    connectivities = adata.obsp[connectivity_key]
    if isinstance(connectivities, np.ndarray):
        coordinates["connectivity"] = connectivities[:, idx]
    else:
        coordinates["connectivity"] = connectivities[:, [idx]].toarray()
    coordinates["y"] = coordinates["y"].max() - coordinates["y"]  # flip y

    p = (
        ggplot(coordinates.sort_values("connectivity", ascending=True), aes(x="x", y="y", colour="connectivity"))
        + geom_point(size=size, shape="8")
        + theme_minimal()
        + labs(colour="connectivity", y="y Coordinate", x="x Coordinate")
        + theme(figure_size=figure_size)
    )

    if return_fig:
        return p

    p.draw()
    return None
