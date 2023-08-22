import numpy as np
from liana.testing._sample_anndata import generate_toy_spatial

from liana.method.sp._lr_bivar import lr_bivar

adata = generate_toy_spatial()


def test_morans_analytical():
    adata = generate_toy_spatial()
    lr_bivar(adata, function_name='morans', n_perms=0, use_raw=True, mask_negatives=True)
    assert 'local_scores' in adata.obsm_keys()
    lrdata = adata.obsm['local_scores']

    assert 'global_res' in lrdata.uns_keys()
    assert 'pvals' in lrdata.layers.keys()

    # test specific interaction
    global_res = lrdata.uns['global_res']
    interaction = global_res[global_res.interaction == 'S100A9^ITGB2']
    np.testing.assert_almost_equal(interaction['global_r'].values, 0.0994394)
    np.testing.assert_almost_equal(interaction['global_pvals'].values, 3.4125671e-07)

    np.testing.assert_almost_equal(np.mean(lrdata[:,'MIF^CD74_CXCR4'].X), -0.008322, decimal=6)
    np.testing.assert_almost_equal(np.mean(lrdata[:,'MIF^CD74_CXCR4'].layers['pvals']), 0.6546573748814639, decimal=6)


def test_cosine_permutation():
    adata = generate_toy_spatial()    
    lr_bivar(adata, function_name='cosine', n_perms=100, use_raw=True)
    lrdata = adata.obsm['local_scores']
    
    assert 'global_res' in lrdata.uns_keys()
    assert 'pvals' in lrdata.layers.keys()
    
    global_res = adata.uns['global_res']
    interaction = global_res[global_res.interaction == 'S100A9^ITGB2']
    
    np.testing.assert_almost_equal(interaction['global_mean'].values, 0.56016606)
    np.testing.assert_almost_equal(interaction['global_sd'].values, 0.33243373)
    
    np.testing.assert_almost_equal(lrdata[:,'MIF^CD74_CXCR4'].X.mean(), 0.32514292, decimal=6)
    np.testing.assert_almost_equal(np.mean(lrdata[:,'MIF^CD74_CXCR4'].layers['pvals']), 0.6274714285714286, decimal=4)


def test_morans_pval_none_cats():
    adata = generate_toy_spatial()
    lr_bivar(adata, function_name='morans', n_perms=None, use_raw=True, add_categories=True)
    assert 'global_res' in adata.uns_keys()
    assert 'local_scores' in adata.obsm_keys()
    # NOT IN
    adata = adata.obsm['local_scores']
    assert 'pvals' not in adata.layers.keys()
    
    assert 'cats' in adata.layers.keys()
    assert adata.layers['cats'].sum() == -6197
