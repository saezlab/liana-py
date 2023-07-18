import anndata as an
from .transform import zi_minmax
from liana.method._pipe_utils._pre import _choose_mtx_rep

def mdata_to_anndata(mdata,
                     x_mod, y_mod,
                     x_layer=None, y_layer=None,
                     x_use_raw=False, y_use_raw=False, 
                     x_transform=False,
                     y_transform=False,
                     cutoff=0.25,
                     verbose=True):
    
    if x_transform is None:
        print('`x_mod` will be transformed to zero-inflated min-max scale.')
        x_transform=lambda x: zi_minmax(x, cutoff=cutoff)
        
    if y_transform is None:
        print('`y_mod` will be transformed to zero-inflated min-max scale.')
        y_transform=lambda x: zi_minmax(x, cutoff=cutoff)
    
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


def _handle_mdata(mdata, x_mod, y_mod, x_layer, y_layer, x_use_raw, y_use_raw, x_transform, y_transform, verbose):
    if x_mod is None or y_mod is None:
        raise ValueError("Both `x_mod` and `y_mod` must be provided!")
    if x_mod not in mdata.mod.keys():
        raise ValueError(f'`x_mod: {x_mod}` is not in the mdata!')
    if y_mod not in mdata.mod.keys():
        raise ValueError(f'`y_mod: {y_mod}` is not in the mdata!')
    
    xdata = mdata.mod[x_mod]
    ydata = mdata.mod[y_mod]
    
    xdata.X = _choose_mtx_rep(xdata, use_raw = x_use_raw, layer = x_layer, verbose=verbose)
    ydata.X = _choose_mtx_rep(ydata, use_raw = y_use_raw, layer = y_layer, verbose=verbose)
    
    # NOTE: if we transform, we copy the data
    if x_transform:
        xdata = xdata.copy()
        xdata.X = x_transform(xdata.X)
    if y_transform:
        ydata = ydata.copy()
        ydata.X = y_transform(ydata.X)
    
    return xdata, ydata
