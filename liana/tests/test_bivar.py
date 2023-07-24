import numpy as np
from itertools import product

from liana.testing._sample_anndata import generate_toy_mdata
from liana.method.sp._bivar import bivar

mdata = generate_toy_mdata()
interactions = list(product(mdata.mod['adata_x'].var.index,
                            mdata.mod['adata_y'].var.index))

def test_bivar_morans():
    # default
    bivar(mdata,
          x_mod='adata_x',
          y_mod='adata_y',
          function_name='morans', 
          interactions=interactions
          )
    assert 'local_scores' in mdata.mod.keys()
    np.testing.assert_almost_equal(mdata.mod['local_scores'].X.sum(), -9.947859, decimal=5)
    
    # with perms
    bivar(mdata, x_mod='adata_x', y_mod='adata_y', 
          function_name='morans', n_perms=2, 
          interactions=interactions)
    
    np.testing.assert_almost_equal(np.mean(mdata.mod['local_scores'].layers['pvals']), 0.604936507, decimal=6)


def test_bivar_nondefault():
    # test different params, inplace = False
    proximity = np.ones((mdata.shape[0], mdata.shape[0]))
    # proximity = np.zeros((mdata.shape[0], mdata.shape[0]))
    # np.fill_diagonal(proximity, 1)
    mdata.obsp['ones'] = proximity
    
    global_stats, local_scores, local_pvals, local_categories = \
          bivar(mdata, x_mod='adata_x', y_mod='adata_y', 
                function_name='morans', n_perms=0,
                connectivity_key='ones', remove_self_interactions=False,
                x_layer = "scaled", y_layer = "scaled", inplace=False, 
                add_categories=True, interactions=interactions
                )
    
    # if all are the same weights, then everything is close to 0?
    np.testing.assert_almost_equal(global_stats['global_r'].sum(), 0)
    local_scores.shape == (700, 100)
    local_pvals.shape == (700, 100)
    np.testing.assert_almost_equal(np.min(np.min(local_pvals)), 0.5, decimal=2)
    
    assert local_categories.sum() == -22400
    
    

def test_bivar_external():
    ones = np.ones((mdata.shape[0], mdata.shape[0]), dtype=np.float64)

    bivar(mdata, x_mod='adata_x',
          y_mod='adata_y',
          function_name='morans', 
          connectivity=ones, interactions=interactions)
    
    
def test_masked_spearman():
    mdata.obsp['spatial_connectivities'] = np.ones([mdata.shape[0], mdata.shape[0]])
    bivar(mdata, x_mod='adata_x', y_mod='adata_y',
          function_name='masked_spearman', interactions=interactions)
    
    # check local
    assert 'local_scores' in mdata.mod.keys()
    np.testing.assert_almost_equal(mdata.mod['local_scores'].X.mean(), 0.22955093, decimal=5)
    
    # check global
    assert 'global_res' in mdata.uns.keys()
    global_res = mdata.uns['global_res']
    assert set(['global_mean','global_sd']).issubset(global_res.columns)
    np.testing.assert_almost_equal(global_res['global_mean'].mean(), 0.229551, decimal=5)
    np.testing.assert_almost_equal(global_res['global_sd'].mean(), 1.0779696e-06, decimal=5)
    

def test_vectorized_pearson():
    # NOTE: something is modifying the mdata object here...
    bivar(mdata, x_mod='adata_x',
          y_mod='adata_y',
          function_name='pearson', n_perms=100,
          interactions=interactions)
    
    # check local
    assert 'local_scores' in mdata.mod.keys()
    adata = mdata.mod['local_scores']
    np.testing.assert_almost_equal(adata.X.mean(), 0.009908355, decimal=5)
    np.testing.assert_almost_equal(adata.layers['pvals'].mean(), 0.754255396825397, decimal=5)
    
    # check global
    assert 'global_res' in mdata.uns.keys()
    global_res = mdata.uns['global_res']
    assert set(['global_mean','global_sd']).issubset(global_res.columns)
    np.testing.assert_almost_equal(global_res['global_mean'].mean(), 0.009908353887100166, decimal=5)
    np.testing.assert_almost_equal(global_res['global_sd'].mean(), 0.3175441555765978, decimal=5)
    
