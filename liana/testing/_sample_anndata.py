import numpy as np
import pandas as pd
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


def generate_anndata(sparsity = 0.90, n_ct = 10, n_vars = 2000, n_obs = 1000, seed=1337):
    # TODO, eventually change completely to use this function, inplace of the other ones
    from scipy.sparse import csr_matrix
    import scanpy as sc
    from liana.utils import spatial_neighbors

    rng = np.random.default_rng(seed=seed)
    counts = rng.poisson(100, size=(n_obs, n_vars))
    mask = rng.choice([0, 1], size=(n_obs, n_vars), p=[sparsity, 1 - sparsity])
    counts = csr_matrix(counts * mask, dtype=np.float32)

    adata = sc.AnnData(counts)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.var_names = [f"Gene{i:d}" for i in range(adata.n_vars)]
    adata.obs_names = [f"Cell{i:d}" for i in range(adata.n_obs)]
    print(f" NNZ fraction: {adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])}")

    x = rng.integers(low=0, high=5000, size=adata.shape[0])
    y = rng.integers(low=0, high=5000, size=adata.shape[0])
    adata.obsm['spatial'] = np.array([x, y]).T

    spatial_neighbors(adata, cutoff=0.1, bandwidth=150, max_neighbours=10)

    # assign cell types
    ct = rng.choice([f"CT{i:d}" for i in range(n_ct)], size=(adata.n_obs,))
    ct = rng.choice(ct, size=(adata.n_obs,))
    adata.obs["cell_type"] = pd.Categorical(ct)

    return adata
