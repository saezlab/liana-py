import numpy as np
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
    np.testing.assert_almost_equal(np.mean(mdata.mod['local_pvals'].X), 0.7872857, decimal=2)
    
    # test different params, inplace = False
    mdata.obsp['ones'] = np.ones((mdata.shape[0], mdata.shape[0]))
    global_stats, local_scores, local_pvals = \
          basis(mdata, x_mod='adata_x', y_mod='adata_y', 
                function_name='morans', pvalue_method="analytical", 
                proximity_key='ones', remove_self_interactions=False,
                x_layer = "scaled", y_layer = "scaled", inplace=False,
                )
    
    # if all are the same weights, then everything is close to 0?
    np.testing.assert_almost_equal(global_stats['global_r'].sum(), 0)
    local_scores.shape == (700, 100)
    local_pvals.shape == (700, 100)
    np.testing.assert_almost_equal(np.min(np.min(local_pvals)), 0.5, decimal=2)
    
    
