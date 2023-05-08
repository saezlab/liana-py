import pandas
from numpy import max, min
from numpy.testing import assert_almost_equal

from liana.method import metalinks
    
from liana.testing._toy_adata import get_toy_adata

# load toy adata
adata = get_toy_adata()

def test_ulm():
    metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', n_perms=10, est_only=True, pass_mask = False)
    assert ('CCC_res' in adata.uns) == False
    assert ('mask' in adata.uns) == False
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)

def test_wsum():
    metalinks(adata, groupby='bulk_labels', est_fun = 'wsum', n_perms=10, est_only=True, pass_mask = False)
    assert ('CCC_res' in adata.uns) == False
    assert ('mask' in adata.uns) == False
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)

def test_wmean():
    metalinks(adata, groupby='bulk_labels', est_fun = 'wmean', n_perms=10, est_only=True, pass_mask = False)
    assert ('CCC_res' in adata.uns) == False
    assert ('mask' in adata.uns) == False
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)

def test_viper():
    metalinks(adata, groupby='bulk_labels', est_fun = 'viper', n_perms=10, est_only=True, pass_mask = False)
    assert ('CCC_res' in adata.uns) == False
    assert ('mask' in adata.uns) == False
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)

# TODO

# def test_udt():
#     metalinks(adata, groupby='bulk_labels', est_fun = 'udt', n_perms=10, est_only=True, pass_mask = False)
#     assert ('CCC_res' in adata.uns) == False
#     assert ('mask' in adata.uns) == False
#     assert adata.obsm['metabolite_abundance'].shape == (700, 29)

# def test_mdt():
#     metalinks(adata, groupby='bulk_labels', est_fun = 'mdt', n_perms=10, est_only=True, pass_mask = False)
#     assert ('CCC_res' in adata.uns) == False
#     assert ('mask' in adata.uns) == False
#     assert adata.obsm['metabolite_abundance'].shape == (700, 29)

# def test_ora():
#     metalinks(adata, groupby='bulk_labels', est_fun = 'ora', n_perms=10, est_only=True, pass_mask = False)
#     assert ('CCC_res' in adata.uns) == False
#     assert ('mask' in adata.uns) == False
#     assert adata.obsm['metabolite_abundance'].shape == (700, 29)

# def test_gsea():
#     metalinks(adata, groupby='bulk_labels', est_fun = 'gsea', n_perms=10, est_only=True, pass_mask = False)
#     assert ('CCC_res' in adata.uns) == False
#     assert ('mask' in adata.uns) == False
#     assert adata.obsm['metabolite_abundance'].shape == (700, 29)

# def test_gsva():
#     metalinks(adata, groupby='bulk_labels', est_fun = 'gsva', n_perms=10, est_only=True, pass_mask = False)
#     assert ('CCC_res' in adata.uns) == False
#     assert ('mask' in adata.uns) == False
#     assert adata.obsm['metabolite_abundance'].shape == (700, 29)



         