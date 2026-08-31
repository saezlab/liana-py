from collections.abc import Callable

import anndata as an
from mudata import MuData

from liana._core._docs import d
from liana._core._pipe_utils._pre import _choose_mtx_rep


@d.dedent
def mdata_to_anndata(mdata: MuData,
                     x_mod: str, y_mod: str,
                     x_layer: str = None, y_layer: str = None,
                     x_use_raw: bool = False, y_use_raw: bool = False,
                     x_transform: Callable = None,
                     y_transform: Callable = None,
                     verbose: bool = True
                     ) -> an.AnnData:
    """
    Convert a MultiData object to an AnnData object.

    Parameters
    ----------
    mdata
        MuData object.
    x_mod
        Name of the modality to be used as x.
    y_mod
        Name of the modality to be used as y.
    x_layer
        Layer to be used for modality x.
    y_layer
        Layer to be used for modality y.
    x_use_raw
        Whether to use raw counts for modality x.
    y_use_raw
        Whether to use raw counts for modality y.
    x_transform
        Transformation function to be applied to modality x.
    y_transform
        Transformation function to be applied to modality y.
    %(verbose)s

    Returns
    -------
    An AnnData object with the two modalities concatenated.
    Information related to observations (obs, obsp, obsm) and `.uns` are copied from the original MuData object.

    Raises
    ------
    ValueError
        If `x_mod` and/or `y_mod` are not provided.

    Examples
    --------
    Joins two modalities of a `MuData` along the variable axis, which is what
    ``liana.mt.bivariate`` expects when relating one modality to another:

    >>> import liana as li
    >>> mdata = li.ds.generate_toy_mdata()
    >>> adata = li.ms.mdata_to_anndata(mdata,
    ...                                x_mod='adata_x',
    ...                                y_mod='adata_y',
    ...                                x_layer='scaled',
    ...                                y_layer='scaled')

    """
    if x_mod is None or y_mod is None:
        raise ValueError("Both `x_mod` and `y_mod` must be provided!")

    xdata = _handle_mod(mdata, x_mod, x_use_raw, x_layer, x_transform, verbose)
    ydata = _handle_mod(mdata, y_mod, y_use_raw, y_layer, y_transform, verbose)

    adata = an.concat([xdata, ydata], axis=1, label='modality')

    adata.obs = mdata.obs.copy()
    adata.obsp = mdata.obsp.copy()
    adata.obsm = mdata.obsm.copy()
    adata.uns = mdata.uns.copy()

    return adata

def _handle_mod(mdata, mod, use_raw, layer, transform, verbose):
    if mod not in mdata.mod.keys():
        raise ValueError(f'`{mod}` is not in the mdata!')

    # NOTE, maybe instead of copying I can just create a minimal AnnData?
    md = mdata.mod[mod].copy()
    if use_raw:
        md = md.raw.to_adata()
    else:
        md.X = _choose_mtx_rep(md, use_raw = use_raw, layer = layer, verbose=verbose)

    if transform:
        if verbose:
            print(f'Transforming {mod} using {transform.__name__}')
        md.X = transform(md.X)
    return md
