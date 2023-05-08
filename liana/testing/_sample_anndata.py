import numpy as np

def generate_toy_spatial():
    from scanpy.datasets import pbmc68k_reduced
    from liana.method.sp._spatial_pipe import spatial_neighbors
    
    adata = pbmc68k_reduced()
    
    rng = np.random.default_rng(seed=1337)
    x = rng.integers(low=0, high=5000, size=adata.shape[0])
    y = rng.integers(low=0, high=5000, size=adata.shape[0])
    adata.obsm['spatial'] = np.array([x, y]).T
    spatial_neighbors(adata, parameter=100, cutoff=0.1)
    
    return adata

def generate_toy_mdata():
    import scanpy as sc
    from mudata import MuData

    adata = generate_toy_spatial()
    adata = adata.raw.to_adata()
    adata = adata[:, 0:10]
    
    adata.layers['scaled'] = sc.pp.scale(adata.X, zero_center=True, max_value=5)
    
    adata_y = adata.copy()
    
    # create mdata
    mdata = MuData({'adata_x':adata, 'adata_y':adata_y})
    mdata.obsp = adata.obsp
    mdata.uns = adata.uns
    mdata.obsm = adata.obsm
    
    return mdata