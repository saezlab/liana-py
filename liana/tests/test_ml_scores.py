import pandas
from numpy import max, min
from numpy.testing import assert_almost_equal

from liana.method import metalinks
    
from liana.testing._toy_adata import get_toy_adata

# load toy adata
adata = get_toy_adata()
expected_shape = adata.shape


def test_cellphone():
    adata = get_toy_adata()
    metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'cellphone', n_perms=10, pass_mask = False)
    
    assert adata.uns['CCC_res'].shape == (2690, 11)
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['CCC_res'].columns

def test_gmean():
    adata = get_toy_adata()
    metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'gmean', n_perms=10, pass_mask = False)

    assert adata.uns['CCC_res'].shape == (2690, 11)
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['CCC_res'].columns

def test_natmi():
    metalinks(adata, groupby='bulk_labels', est_fun = 'ulm', score_fun = 'natmi', n_perms=10, pass_mask = False)
    
    assert adata.uns['CCC_res'].shape == (2690, 13)
    assert adata.obsm['metabolite_abundance'].shape == (700, 29)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['CCC_res'].columns






         