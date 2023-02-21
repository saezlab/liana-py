import numpy as np
from liana.testing._sample_anndata import generate_toy_mdata

from liana.method.sp._spatial_pipe import basis

def test_morans():
    mdata = generate_toy_mdata()    
    
    basis(mdata, x_mod='adata_x', y_mod='adata_y', function_name='morans')
    
    mdata.obsp['ones'] = np.ones((mdata.shape[0], mdata.shape[0]))
    # with weird params
    basis(mdata, x_mod='adata_x', y_mod='adata_y', 
          function_name='morans', pvalue_method="analytical", 
          proximity_key='ones', remove_self_interactions=False,
          x_layer = "scaled", y_layer = "scaled")
    # if all are the same weights, then everything is close to 0?
    np.testing.assert_almost_equal(mdata.obsm['local_score']['HES4&HES4'][0], 0)
    
    # with perms
    basis(mdata, x_mod='adata_x', y_mod='adata_y', 
          function_name='morans', pvalue_method="permutation")
    
