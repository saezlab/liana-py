import numpy as np
from scanpy.datasets import pbmc68k_reduced
from  scipy.sparse import csr_matrix

from liana.method.sp._spatialdm import spatialdm
from liana.method.sp._spatial_utils import _global_zscore_pvals, _global_permutation_pvals, _local_permutation_pvals, _local_zscore_pvals
from liana.method.sp._bivariate_funs import _local_morans


adata = pbmc68k_reduced()
proximity = np.zeros([adata.shape[0], adata.shape[0]])
np.fill_diagonal(proximity, 1)
adata.obsm['proximity'] = proximity


# toy test data
seed = 0
rng = np.random.default_rng(seed=seed)
dist = csr_matrix(rng.normal(size=(10, 10)))
x_mat = rng.normal(size=(10, 10))
y_mat = rng.normal(size=(10, 10))
n_perm = 100
positive_only = True
    

def test_global_permutation_pvals():
    global_truth = rng.normal(size=(10))

    pvals = _global_permutation_pvals(x_mat=x_mat,
                                      y_mat=y_mat,
                                      global_r=global_truth,
                                      seed=seed,  
                                      n_perm=n_perm,
                                      positive_only=positive_only,
                                      dist = dist)
    assert pvals.shape == (10, )
    assert pvals.sum().round(3)==4.65
    
    

def test_local_permutation_pvals():
    local_truth = rng.normal(size=(10, 10))
    positive_only = True

    pvals = _local_permutation_pvals(x_mat = x_mat,
                                     y_mat = y_mat,
                                     local_truth = local_truth,
                                     local_fun = _local_morans,
                                     dist = dist,
                                     n_perm = n_perm,
                                     seed = seed,
                                     positive_only=positive_only)
    assert pvals.shape == (10, 10)
        


def test_global_zscore_pvals():
    global_truth = rng.normal(size=(10))
    pvals = _global_zscore_pvals(global_r=global_truth, dist=dist, positive_only=positive_only)
    assert pvals.shape == (10, )


def test_local_zscore_pvals():
    local_truth = rng.normal(size=(10, 10))
    pvals = _local_zscore_pvals(x_mat=x_mat, y_mat=y_mat, dist=dist, local_r=local_truth, positive_only=positive_only)
    assert pvals.shape == (10, 10)



def test_spatialdm():
    spatialdm(adata, use_raw=True)
    assert 'global_res' in adata.uns_keys()
    assert 'local_r' in adata.obsm_keys()
    assert 'local_pvals' in adata.obsm_keys()

    # test specific interaction
    global_res = adata.uns['global_res']
    interaction = global_res[global_res.interaction == 'S100A9&ITGB2']
    np.testing.assert_almost_equal(interaction['global_r'].values, 0.3100374)
    np.testing.assert_almost_equal(interaction['global_pvals'].values, 1.232729e-16)

    assert np.mean(adata.obsm['local_r']['MIF&CD74_CXCR4']) == 0.024113696644942034
    assert np.mean(adata.obsm['local_pvals']['TNFSF13B&TNFRSF13B']) == 0.9214972206911888


def test_spatialdm_permutation():
    spatialdm(adata, use_raw=True, pvalue_method="permutation")
    assert 'global_res' in adata.uns_keys()
    assert 'local_r' in adata.obsm_keys()
    assert 'local_pvals' in adata.obsm_keys()
    
    global_res = adata.uns['global_res']
    interaction = global_res[global_res.interaction == 'S100A9&ITGB2']
    
    np.testing.assert_almost_equal(interaction['global_r'].values, 0.3100374)
    np.testing.assert_almost_equal(interaction['global_pvals'].values, 0.0)
    
    assert np.mean(adata.obsm['local_r']['MIF&CD74_CXCR4']) == 0.024113696644942034
    assert np.mean(adata.obsm['local_pvals']['TNFSF13B&TNFRSF13B']) == 0.9874585714285714
