import anndata as an
from .transform import zi_minmax

def mdata_to_anndata(mdata, mod_x, mod_y, transform=True):
    if mod_x is None or mod_y is None:
        raise ValueError("Both `mod_x` and `mod_y` must be provided!")
    if mod_x not in mdata.mod.keys():
        raise ValueError(f'`mod_x: {mod_x}` is not in the mdata!')
    if mod_y not in mdata.mod.keys():
        raise ValueError(f'`mod_y: {mod_y}` is not in the mdata!')
    
    mdx = mdata.mod[mod_x]
    mdy = mdata.mod[mod_y]
    
    adata = an.concat([mdx, mdy], join='outer', axis=1, merge='first', label='modality')
    
    if transform is None:
        print('Transforming data to zero-inflated min-max scale')
        transform = True
    if transform:
        adata.X = zi_minmax(adata.X, cutoff=0.25) # TODO add cutoff as a parameter
    
    return adata
    
