import anndata as an
from liana.method._pipe_utils._pre import _choose_mtx_rep

def mdata_to_anndata(mdata,
                     x_mod, y_mod,
                     x_layer=None, y_layer=None,
                     x_use_raw=False, y_use_raw=False, 
                     x_transform=None,
                     y_transform=None,
                     cutoff=None,
                     verbose=True
                     ):

    xdata, ydata = _handle_mdata(mdata, 
                                 x_mod, y_mod,
                                 x_layer, y_layer,
                                 x_use_raw, y_use_raw,
                                 x_transform=x_transform,
                                 y_transform=y_transform,
                                 verbose=verbose,
                                 )
    
    adata = an.concat([xdata, ydata], join='outer', axis=1, merge='first', label='modality')
    
    return adata


def _handle_mdata(mdata, 
                  x_mod, y_mod,
                  x_layer, y_layer,
                  x_use_raw, y_use_raw,
                  x_transform, y_transform, 
                  verbose):
    if x_mod is None or y_mod is None:
        raise ValueError("Both `x_mod` and `y_mod` must be provided!")
    
    xdata = _handle_mod(mdata, x_mod, x_use_raw, x_layer, x_transform, verbose)
    ydata = _handle_mod(mdata, y_mod, y_use_raw, y_layer, y_transform, verbose)
    
    return xdata, ydata

def _handle_mod(mdata, mod, use_raw, layer, transform, verbose):
    if mod not in mdata.mod.keys():
        raise ValueError(f'`{mod}` is not in the mdata!')
    
    md = mdata.mod[mod]
    md.X = _choose_mtx_rep(md, use_raw = use_raw, layer = layer, verbose=verbose)
    
    if transform:
        if verbose:
            print(f'Transforming {mod} using {transform.__name__}')
        md = md.copy()
        md.X = transform(md.X)
    return md