import pandas
from numpy import max, min
from numpy.testing import assert_almost_equal

from liana.method import metalinks
    
from liana.testing._toy_adata import get_toy_adata

# load toy adata
adata = get_toy_adata()
expected_shape = adata.shape


def test_cellphone():
    metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'cellphone', n_perms=10, pass_mask = False)
    
    assert adata.uns['CCC_res'].shape == (100, 11)
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['CCC_res'].columns

def test_gmean():
    metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'gmean', n_perms=10, pass_mask = False)

    assert adata.uns['CCC_res'].shape == (100, 11)
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['CCC_res'].columns

def test_natmi():
    metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'natmi', n_perms=10, pass_mask = False)
    
    assert adata.uns['CCC_res'].shape == (100, 13)
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['CCC_res'].columns

# def test_cellchat():
#     metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'cellchat', n_perms=10, pass_mask = False)
    
#     assert adata.uns['CCC_res'].shape == (100, 11)
#     assert adata.obsm['metabolite_abundance'].shape == (700, 29)
#     assert ('mask' in adata.uns) == False
#     assert 'ligand_name' in adata.uns['CCC_res'].columns

# def test_connectome():
#     metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'connectome', n_perms=10, pass_mask = False)
    
#     assert adata.uns['CCC_res'].shape == (100, 11)
#     assert adata.obsm['metabolite_abundance'].shape == (700, 29)
#     assert ('mask' in adata.uns) == False
#     assert 'ligand_name' in adata.uns['CCC_res'].columns



# def test_logfc():
#     metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'logfc', n_perms=10, pass_mask = False)
    
#     assert adata.uns['CCC_res'].shape == (100, 11)
#     assert adata.obsm['metabolite_abundance'].shape == (700, 29)
#     assert ('mask' in adata.uns) == False
#     assert 'ligand_name' in adata.uns['CCC_res'].columns


# def test_sca():
#     metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'sca', n_perms=10, pass_mask = False)
    
#     assert adata.uns['CCC_res'].shape == (100, 11)
#     assert adata.obsm['metabolite_abundance'].shape == (700, 29)
#     assert ('mask' in adata.uns) == False
#     assert 'ligand_name' in adata.uns['CCC_res'].columns





         