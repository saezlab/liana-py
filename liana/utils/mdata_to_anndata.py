import anndata as an
from liana.method._pipe_utils._pre import _choose_mtx_rep
from liana._docs import d

@d.dedent
def mdata_to_anndata(mdata,
                     x_mod, y_mod,
                     x_layer=None, y_layer=None,
                     x_use_raw=False, y_use_raw=False,
                     x_transform=None,
                     y_transform=None,
                     verbose=True
                     ):
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
    md.X = _choose_mtx_rep(md, use_raw = use_raw, layer = layer, verbose=verbose)

    if transform:
        if verbose:
            print(f'Transforming {mod} using {transform.__name__}')
        md.X = transform(md.X)
    return md
