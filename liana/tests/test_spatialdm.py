import numpy as np
from scanpy.datasets import pbmc68k_reduced

from liana.method.sp._spatialdm import spatialdm, _global_zscore_pvals, _global_permutation_pvals, \
    _local_zscore_pvals, _local_permutation_pvals

adata = pbmc68k_reduced()
proximity = np.zeros([adata.shape[0], adata.shape[0]])
np.fill_diagonal(proximity, 1)
adata.obsm['proximity'] = proximity


def test_global_permutation_pvals():
    1


def test_global_zscore_pvals():
    1


def test_local_zscore_pvals():
    1


def test_local_permutation_pvals():
    1


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
    assert np.mean(adata.obsm['local_pvals']['TNFSF13B&TNFRSF13B']) == 0.9214972220246149
