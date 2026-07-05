from __future__ import annotations

import numpy as np
from anndata import AnnData

from liana._constants import DefaultValues as V
from liana._constants import Keys as K
from liana._docs import d
from liana._logging import _logg


def _grid_offsets(extents: np.ndarray, n_cols: int, margin: float) -> np.ndarray:
    """
    Assign each sample a non-overlapping offset on a fixed-column grid.

    Parameters
    ----------
    extents
        Array of shape (n_samples, 2) with the (width, height) spanned by each sample.
    n_cols
        Number of columns in the grid.
    margin
        Fractional margin added to the largest sample's extent, used as the grid's cell size.

    Returns
    -------
    Array of shape (n_samples, 2) with the (x, y) offset assigned to each sample.

    """
    cell_size = extents.max(axis=0) * (1 + margin)
    rows, cols = np.divmod(np.arange(extents.shape[0]), n_cols)

    return np.column_stack([cols, rows]) * cell_size


@d.dedent
def _expand_coordinates(adatas: list[AnnData],
                        spatial_key: str = K.spatial_key,
                        n_cols: int = 4,
                        margin: float = 0.2,
                        verbose: bool = V.verbose
                        ) -> list[AnnData]:
    """
    Lay out the spatial coordinates of multiple samples onto a non-overlapping grid.

    Samples profiled independently (e.g. separate Visium slides) typically share
    overlapping coordinate ranges. This translates each sample onto its own cell of a
    `n_cols`-wide grid, so that samples can be jointly visualized or processed without
    their coordinates colliding.

    Parameters
    ----------
    adatas
        List of AnnData objects, each containing spatial coordinates in `.obsm[spatial_key]`.
    %(spatial_key)s
    n_cols
        Number of columns in the grid onto which samples are arranged.
    margin
        Fractional margin added between adjacent samples, relative to the largest
        sample's extent, to prevent them from touching. Default is `0.2`, i.e. a 20%% margin.
    %(verbose)s

    Returns
    -------
    A list of new AnnData objects (copies of `adatas`), with non-overlapping coordinates
    written to `.obsm[spatial_key]`. The original coordinates are preserved in
    `.obsm[f'{spatial_key}_original']`.

    """
    coords = [adata.obsm[spatial_key] for adata in adatas]
    extents = np.array([coord.max(axis=0) - coord.min(axis=0) for coord in coords])

    offsets = _grid_offsets(extents=extents, n_cols=n_cols, margin=margin)

    _logg(f"Expanding coordinates of {len(adatas)} samples onto a {n_cols}-column grid.", verbose=verbose)

    expanded = []
    for adata, coord, offset in zip(adatas, coords, offsets):
        adata = adata.copy()
        adata.obsm[f'{spatial_key}_original'] = coord.copy()
        adata.obsm[spatial_key] = coord - coord.min(axis=0) + offset
        expanded.append(adata)

    return expanded
