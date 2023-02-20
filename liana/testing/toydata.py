import numpy as np

def generate_toy_spatial():
    from scanpy.datasets import pbmc68k_reduced
    from liana.method.sp._spatial_utils import get_spatial_proximity
    
    adata = pbmc68k_reduced()
    
    rng = np.random.default_rng(seed=1337)
    x = rng.integers(low=0, high=5000, size=adata.shape[0])
    y = rng.integers(low=0, high=5000, size=adata.shape[0])
    adata.obsm['spatial'] = np.array([x, y]).T
    get_spatial_proximity(adata, parameter=100, cutoff=0.1)
    
    return adata