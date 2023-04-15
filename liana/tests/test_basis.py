import numpy as np
import pandas as pd
from itertools import product

from liana.testing._sample_anndata import generate_toy_mdata
from liana.method.sp._basis import basis

def test_morans():
    mdata = generate_toy_mdata()    
    
    # default
    basis(mdata, x_mod='adata_x', y_mod='adata_y', function_name='morans')
    assert 'local_scores' in mdata.mod.keys()
    np.testing.assert_almost_equal(mdata.mod['local_scores'].X.sum(), -124.05389, decimal=5)
    
    # with perms
    basis(mdata, x_mod='adata_x', y_mod='adata_y', 
          function_name='morans', pvalue_method="permutation", n_perms=2)
    np.testing.assert_almost_equal(np.mean(mdata.obsm['local_pvals'].values), 0.7872857, decimal=6)


def test_basis_nondefault():
    mdata = generate_toy_mdata()   
    # test different params, inplace = False
    proximity = np.ones((mdata.shape[0], mdata.shape[0]))
    # proximity = np.zeros((mdata.shape[0], mdata.shape[0]))
    # np.fill_diagonal(proximity, 1)
    mdata.obsp['ones'] = proximity
    
    global_stats, local_scores, local_pvals = \
          basis(mdata, x_mod='adata_x', y_mod='adata_y', 
                function_name='morans', pvalue_method="analytical", 
                proximity_key='ones', remove_self_interactions=False,
                x_layer = "scaled", y_layer = "scaled", inplace=False, 
                add_categories=True
                )
    
    # if all are the same weights, then everything is close to 0?
    np.testing.assert_almost_equal(global_stats['global_r'].sum(), 0)
    local_scores.shape == (700, 100)
    local_pvals.shape == (700, 100)
    np.testing.assert_almost_equal(np.min(np.min(local_pvals)), 0.5, decimal=2)
    
    categories = mdata.obsm['local_categories']
    assert categories.values.sum() == -22400
    
    

def test_basis_external():
    mdata = generate_toy_mdata()
    ones = np.ones((mdata.shape[0], mdata.shape[0]), dtype=np.float64)
    
    x_vars = mdata.mod['adata_x'].var.index[-3:]
    y_vars = mdata.mod['adata_y'].var.index[0:3]
    interactions = list(product(x_vars, y_vars))
    
    basis(mdata, x_mod='adata_x', y_mod='adata_y', function_name='morans', proximity=ones, interactions=interactions)
    
    
def test_masked_pearson():
    mdata = generate_toy_mdata()
    basis(mdata, x_mod='adata_x', y_mod='adata_y', function_name='masked_pearson')
    
    # check local
    assert 'local_scores' in mdata.mod.keys()
    mdata.mod['local_scores'].X[np.isnan(mdata.mod['local_scores'].X)]=0 # TODO fix this
    np.testing.assert_almost_equal(mdata.mod['local_scores'].X.mean(), 0.010296326, decimal=5)
    
    # check global
    assert 'global_res' in mdata.uns.keys()
    assert set(['global_mean','global_sd']).issubset(mdata.uns['global_res'].columns)
    # check specific values are what we expect
    

def test_vectorized_pearson():
    mdata = generate_toy_mdata()
    basis(mdata, x_mod='adata_x', y_mod='adata_y', function_name='pearson')
    
    # check local
    assert 'local_scores' in mdata.mod.keys()
    np.testing.assert_almost_equal(mdata.mod['local_scores'].X.mean(), 0.009908355, decimal=5)
    
    # check global
    assert 'global_res' in mdata.uns.keys()
    assert set(['global_mean','global_sd']).issubset(mdata.uns['global_res'].columns)
    # check specific values are what we expect