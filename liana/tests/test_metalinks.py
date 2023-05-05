import pandas
from numpy import max, min
from numpy.testing import assert_almost_equal

from liana.method import metalinks
    
from liana.testing._toy_adata import get_toy_adata

# load toy adata
adata = get_toy_adata()


def test_metalinks_default():
    metalinks(adata, groupby='bulk_labels', n_perms=10)
    
    assert adata.uns['CCC_res'].shape == (100, 11)
    assert adata.obsm['metabolite_abundance'].shape == (700, 421)
    assert adata.uns['mask'].shape == (765, 421)
    assert adata.uns['mask'].sum().sum() == -184
    assert 'ligand_name' in adata.uns['CCC_res'].columns

def test_metalinks_without_mask():
    metalinks(adata, groupby='bulk_labels', n_perms=10, pass_mask = False)
    
    assert adata.uns['CCC_res'].shape == (100, 11)
    assert adata.obsm['metabolite_abundance'].shape == (700, 421)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['CCC_res'].columns

def test_metalinks_est_only():
    metalinks(adata, groupby='bulk_labels', n_perms=10, est_only=True)
    
    assert ('CCC_res' in adata.uns) == False
    assert adata.obsm['metabolite_abundance'].shape == (700, 421)
    assert adata.uns['mask'].shape == (765, 421)
    assert adata.uns['mask'].sum().sum() == -184


def test_metalinks_est_only_without_mask():
    metalinks(adata, groupby='bulk_labels', n_perms=10, est_only=True, pass_mask = False)

    assert ('CCC_res' in adata.uns) == False
    assert ('mask' in adata.uns) == False
    assert adata.obsm['metabolite_abundance'].shape == (700, 421)
         