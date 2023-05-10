import numpy as np
from liana.testing._sample_anndata import generate_toy_spatial

from liana.method.sp._lr_basis import lr_basis

adata = generate_toy_spatial()    


def test_morans_analytical():
    adata = generate_toy_spatial()
    lr_basis(adata, function_name='morans', pvalue_method="analytical", use_raw=True)
    assert 'global_res' in adata.uns_keys()
    assert 'local_scores' in adata.obsm_keys()
    assert 'local_pvals' in adata.obsm_keys()

    # test specific interaction
    global_res = adata.uns['global_res']
    interaction = global_res[global_res.interaction == 'S100A9&ITGB2']
    np.testing.assert_almost_equal(interaction['global_r'].values, 0.0994394)
    np.testing.assert_almost_equal(interaction['global_pvals'].values, 3.4125671e-07)

    np.testing.assert_almost_equal(np.mean(adata.obsm['local_scores']['MIF&CD74_CXCR4']), -0.0013958386, decimal=6)
    np.testing.assert_almost_equal(np.mean(adata.obsm['local_pvals']['TNFSF13B&TNFRSF13B']), 0.8878231110397902, decimal=6)


def test_morans_permutation():
    adata = generate_toy_spatial()    
    lr_basis(adata, function_name='morans', pvalue_method="permutation", use_raw=True)
    assert 'global_res' in adata.uns_keys()
    assert 'local_scores' in adata.obsm_keys()
    assert 'local_pvals' in adata.obsm_keys()
    
    global_res = adata.uns['global_res']
    interaction = global_res[global_res.interaction == 'S100A9&ITGB2']
    
    np.testing.assert_almost_equal(interaction['global_r'].values, 0.0994394)
    np.testing.assert_almost_equal(interaction['global_pvals'].values, 0.0)
    
    np.testing.assert_almost_equal(np.mean(adata.obsm['local_scores']['MIF&CD74_CXCR4']), -0.0013958386, decimal=6)
    np.testing.assert_almost_equal(np.mean(adata.obsm['local_pvals']['TNFSF13B&TNFRSF13B']), 0.9419328571428572, decimal=6)


def test_morans_pval_none_cats():
    adata = generate_toy_spatial()
    lr_basis(adata, function_name='morans', pvalue_method=None, use_raw=True, add_categories=True)
    assert 'global_res' in adata.uns_keys()
    assert 'local_scores' in adata.obsm_keys()
    # NOT IN
    assert 'local_pvals' not in adata.obsm_keys()
    
    assert 'local_categories' in adata.obsm_keys()
    assert adata.obsm['local_categories'].values.sum() == -10306
