import pandas
from numpy import max, min
from numpy.testing import assert_almost_equal

from liana.method import metalinks
    
from liana.testing._toy_adata import get_toy_adata

# load toy adata
adata = get_toy_adata()


def test_metalinks_est_only_without_mask():
    metalinks(adata, groupby='bulk_labels', n_perms=10, est_only=True, pass_mask = False)

    assert ('CCC_res' in adata.uns) == False
    assert ('mask' in adata.uns) == False
    assert adata.obsm['metabolite_abundance'].shape == (700, 230)

def test_metalinks_without_mask():
    metalinks(adata, groupby='bulk_labels', n_perms=10, pass_mask = False)
    
    assert adata.uns['CCC_res'].shape == (5737, 11)
    assert adata.obsm['metabolite_abundance'].shape == (700, 230)
    assert ('mask' in adata.uns) == False
    assert 'ligand_name' in adata.uns['CCC_res'].columns

def test_metalinks_est_only():
    adata = get_toy_adata()
    metalinks(adata, groupby='bulk_labels', n_perms=10, est_only=True)
    
    assert ('CCC_res' in adata.uns) == False
    assert adata.obsm['metabolite_abundance'].shape == (700, 230)
    assert adata.uns['mask'].shape == (230, 765)
    assert adata.uns['mask'].sum().sum() == 181

def test_metalinks_default():
    metalinks(adata, groupby='bulk_labels', n_perms=10)
    
    assert adata.uns['CCC_res'].shape == (5737, 11)
    assert adata.obsm['metabolite_abundance'].shape == (700, 230)
    assert adata.uns['mask'].shape == (765, 230)
    assert adata.uns['mask'].sum().sum() == 181
    assert 'ligand_name' in adata.uns['CCC_res'].columns

def test_metalinks_transporter_est_only_without_mask():
    adata = get_toy_adata()
    metalinks(adata, groupby='bulk_labels', n_perms=10, metsets_name='transport',
           est_fun='transport', score_fun= 'gmean', min_cells=3, pass_mask=False, est_only=True)
         
    assert ('CCC_res' in adata.uns) == False
    assert ('mask' in adata.uns) == False
    assert ('efflux_mask' in adata.uns) == False
    assert ('influx_mask' in adata.uns) == False
    assert 'metabolite_abundance' in adata.obsm
    assert 'efflux' in adata.obsm
    assert 'influx' in adata.obsm
    assert adata.obsm['metabolite_abundance'].shape == (700, 16)
    assert adata.obsm['efflux'].shape == (700, 1)
    assert adata.obsm['influx'].shape == (700, 1)
    assert_almost_equal(adata.obsm['efflux'].sum().sum(),  350.08, decimal=2)
    assert_almost_equal(adata.obsm['influx'].sum().sum(),  350.08, decimal=2)


def test_metalinks_transporter_without_mask():
    adata = get_toy_adata()
    metalinks(adata, groupby='bulk_labels', n_perms=10, metsets_name='transport',
           est_fun='transport', score_fun= 'gmean', min_cells=3, pass_mask=False)
    
    assert 'CCC_res' in adata.uns
    assert ('mask' in adata.uns) == False
    assert ('efflux_mask' in adata.uns) == False
    assert ('influx_mask' in adata.uns) == False
    assert 'metabolite_abundance' in adata.obsm
    assert 'efflux' in adata.obsm
    assert 'influx' in adata.obsm
    assert adata.obsm['metabolite_abundance'].shape == (700, 16)
    assert adata.obsm['efflux'].shape == (700, 1)
    assert adata.obsm['influx'].shape == (700, 1)
    assert_almost_equal(adata.obsm['efflux'].sum().sum(),  350.08, decimal=2)
    assert_almost_equal(adata.obsm['influx'].sum().sum(),  350.08, decimal=2)


def test_metalinks_transporter_est_only():
    
    adata = get_toy_adata()
    metalinks(adata, groupby='bulk_labels', n_perms=10, metsets_name='transport',
           est_fun='transport', score_fun= 'gmean', min_cells=3, pass_mask=True, est_only=True)
    
    assert ('CCC_res' in adata.uns) == False
    assert 'mask' in adata.uns
    assert 'efflux_mask' in adata.uns
    assert 'influx_mask' in adata.uns
    assert 'metabolite_abundance' in adata.obsm
    assert 'efflux' in adata.obsm
    assert 'influx' in adata.obsm
    assert adata.obsm['metabolite_abundance'].shape == (700, 16)
    assert adata.obsm['efflux'].shape == (700, 1)
    assert adata.obsm['influx'].shape == (700, 1)
    assert adata.uns['mask'].shape == (16, 765)
    assert adata.uns['mask'].sum().sum() == -57
    assert_almost_equal(adata.obsm['efflux'].sum().sum(),  350.08, decimal=2)
    assert_almost_equal(adata.obsm['influx'].sum().sum(),  350.08, decimal=2)
    assert adata.uns['efflux_mask'].sum().sum() == 1
    assert adata.uns['influx_mask'].sum().sum() == -3



def test_metalinks_transporter_default():
    metalinks(adata, groupby='bulk_labels', n_perms=10, metsets_name='transport',
           est_fun='transport', score_fun= 'gmean', min_cells=3)
    
    assert 'CCC_res' in adata.uns
    assert 'mask' in adata.uns
    assert 'efflux_mask' in adata.uns
    assert 'influx_mask' in adata.uns
    assert 'metabolite_abundance' in adata.obsm
    assert 'efflux' in adata.obsm
    assert 'influx' in adata.obsm
    assert adata.obsm['metabolite_abundance'].shape == (700, 16)
    assert adata.obsm['efflux'].shape == (700, 1)
    assert adata.obsm['influx'].shape == (700, 1)
    assert adata.uns['mask'].shape == (16, 765)
    assert adata.uns['mask'].sum().sum() == -57
    assert_almost_equal(adata.obsm['efflux'].sum().sum(),  350.08, decimal=2)
    assert_almost_equal(adata.obsm['influx'].sum().sum(),  350.08, decimal=2)
    assert adata.uns['efflux_mask'].sum().sum() == 1
    assert adata.uns['influx_mask'].sum().sum() == -3