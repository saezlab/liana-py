import numpy as np
from scanpy.datasets import pbmc68k_reduced
from liana.method import get_spatial_proximity

# TODO move all of this repetitive code to a single function in testing
adata = pbmc68k_reduced()
proximity = np.zeros([adata.shape[0], adata.shape[0]])

rng = np.random.default_rng(seed=1337)
x = rng.integers(low=0, high=5000, size=adata.shape[0])
y = rng.integers(low=0, high=5000, size=adata.shape[0])
adata.obsm['spatial'] = np.array([x, y]).T
get_spatial_proximity(adata, parameter=100, cutoff=0.1)


def test_get_spatial_proximity():
    get_spatial_proximity(adata=adata, parameter=200, bypass_diagonal=False, cutoff=0.2)

