from liana.method import metalinks
from liana.testing._toy_adata import get_toy_adata
from numpy.testing import assert_array_equal

# load toy adata
adata = get_toy_adata()
expected_shape = adata.shape

def test_cellphone():
    adata = get_toy_adata()
    metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'cellphone', n_perms=10, pass_mask = False)
    
    assert adata.uns['liana_res'].shape == (2690, 11)
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['liana_res'].columns

def test_gmean():
    adata = get_toy_adata()
    metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'gmean', n_perms=10, pass_mask = False)

    assert adata.uns['liana_res'].shape == (2690, 11)
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['liana_res'].columns
    assert_array_equal(adata.uns['liana_res']['metalinks_score'].mean(), 0.6172697517840214)
    assert_array_equal(adata.uns['liana_res']['pval'].mean(), 0.4804460966542751)
    

def test_natmi():
    metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'natmi', n_perms=10, pass_mask = False)
    
    assert adata.uns['liana_res'].shape == (2690, 13)
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['liana_res'].columns






         