import numpy as np
from scanpy.datasets import pbmc68k_reduced

from liana.method.sp._spatialdm import spatialdm

adata = pbmc68k_reduced()
proximity = np.zeros([adata.shape[0], adata.shape[0]])
np.fill_diagonal(proximity, 1)
adata.obsm['proximity'] = proximity


def test_spatialdm():
    spatialdm(adata, use_raw=True)
    assert 'global_res' in adata.uns_keys()
    assert 'local_r' in adata.obsm_keys()
    assert 'local_pvals' in adata.obsm_keys()


def test_global_permutation_pvals():
    1


def test_global_zscore_pvals():
    1


def test_local_zscore_pvals():
    1


def test_local_permutation_pvals():
    1


