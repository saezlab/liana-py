import numpy as np
from  scipy.sparse import csr_matrix

from scanpy.datasets import pbmc68k_reduced
from liana.method import get_spatial_proximity

from liana.method.sp._spatial_utils import _global_zscore_pvals, _global_permutation_pvals, _local_permutation_pvals, _local_zscore_pvals
from liana.method.sp._bivariate_funs import _local_morans

from liana.testing.toydata import generate_toy_spatial

adata = generate_toy_spatial()

def test_get_spatial_proximity():
    get_spatial_proximity(adata=adata, parameter=200, bypass_diagonal=False, cutoff=0.2)
    np.testing.assert_equal(adata.obsm['proximity'].shape, (adata.shape[0], adata.shape[0]))
    np.testing.assert_equal(adata.obsm['proximity'].sum(), 4550.654013895928)
    
    get_spatial_proximity(adata=adata, parameter=100, bypass_diagonal=False, cutoff=0.1)
    np.testing.assert_equal(adata.obsm['proximity'].sum(), 1802.332962418902)
    

# toy test data
seed = 0
rng = np.random.default_rng(seed=seed)
dist = csr_matrix(rng.normal(size=(10, 10)))

norm_factor = dist.shape[0] / dist.sum()
weight = csr_matrix(norm_factor * dist)

x_mat = rng.normal(size=(10, 10))
y_mat = rng.normal(size=(10, 10))
n_perms = 100
positive_only = True
    

def test_global_permutation_pvals():
    global_truth = rng.normal(size=(10))

    pvals = _global_permutation_pvals(x_mat=x_mat,
                                      y_mat=y_mat,
                                      global_r=global_truth,
                                      seed=seed,  
                                      n_perms=n_perms,
                                      positive_only=positive_only,
                                      weight = weight)
    assert pvals.shape == (10, )
    assert pvals.sum().round(3)==4.65
    
    

def test_local_permutation_pvals():
    local_truth = rng.normal(size=(10, 10))
    positive_only = True

    pvals = _local_permutation_pvals(x_mat = x_mat,
                                     y_mat = y_mat,
                                     local_truth = local_truth,
                                     local_fun = _local_morans,
                                     weight = weight,
                                     n_perms = n_perms,
                                     seed = seed,
                                     positive_only=positive_only)
    assert pvals.shape == (10, 10)


def test_global_zscore_pvals():
    global_truth = rng.normal(size=(10))
    pvals = _global_zscore_pvals(global_r=global_truth, weight=weight, positive_only=positive_only)
    assert pvals.shape == (10, )


def test_local_zscore_pvals():
    local_truth = rng.normal(size=(10, 10))
    pvals = _local_zscore_pvals(x_mat=x_mat, y_mat=y_mat, weight=weight, local_truth=local_truth, positive_only=positive_only)
    assert pvals.shape == (10, 10)
