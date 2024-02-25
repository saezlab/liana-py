import numpy as np
from numpy import random
from scanpy.datasets import pbmc68k_reduced
from liana.utils.spatial_neighbors import spatial_neighbors

def generate_toy_spatial():
    adata = pbmc68k_reduced()

    rng = np.random.default_rng(seed=1337)
    x = rng.integers(low=0, high=5000, size=adata.shape[0])
    y = rng.integers(low=0, high=5000, size=adata.shape[0])
    adata.obsm['spatial'] = np.array([x, y]).T
    spatial_neighbors(adata, bandwidth=100, cutoff=0.1, set_diag=True)

    return adata

def generate_toy_mdata():
    import scanpy as sc
    from mudata import MuData

    adata = generate_toy_spatial()
    adata = adata.raw.to_adata()
    adata = adata[:, 0:10]
    sc.pp.filter_cells(adata, min_counts=1)

    adata.layers['scaled'] = sc.pp.scale(adata.X, zero_center=True, max_value=5)

    adata_y = adata.copy()

    # create mdata
    mdata = MuData({'adata_x':adata, 'adata_y':adata_y})
    mdata.obsp = adata.obsp
    mdata.uns = adata.uns
    mdata.obsm = adata.obsm
    mdata.obs = adata.obs

    return mdata


def generate_toy_adata():
    adata = pbmc68k_reduced()
    sample_key = 'sample'

    rng = random.default_rng(0)

    # create fake samples
    adata.obs[sample_key] = rng.choice(['A', 'B', 'C', 'D'], size=len(adata.obs))

    # group samples into conditions
    adata.obs['case'] = adata.obs[sample_key].map({'A': 'yes', 'B': 'yes', 'C': 'no', 'D': 'no'})

    return adata
