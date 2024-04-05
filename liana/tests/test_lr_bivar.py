import numpy as np
from liana.testing._sample_anndata import generate_toy_spatial

from liana.method.sp._bivariate._lr_bivar import lr_bivar

adata = generate_toy_spatial()
 # NOTE: these should be the same regardless of the local function
expected_gmorans = 0.0994394
expected_glee = 0.04854206

def test_morans_analytical():
    adata = generate_toy_spatial()
    lr_bivar(adata,
             local_name='morans',
             global_name=['morans'],
             n_perms=0, # NOTE issue with Lee here
             use_raw=True,
             mask_negatives=True)
    assert 'local_scores' in adata.obsm_keys()
    lrdata = adata.obsm['local_scores']

    assert 'pvals' in lrdata.layers.keys()

    np.testing.assert_almost_equal(np.mean(lrdata[:,'MIF^CD74_CXCR4'].X), 0.12803833, decimal=6)
    np.testing.assert_almost_equal(np.mean(lrdata[:,'MIF^CD74_CXCR4'].layers['pvals']), 0.8764923, decimal=6)

    interaction = lrdata.var[lrdata.var.index == 'S100A9^ITGB2']
    np.testing.assert_almost_equal(interaction['morans'].values, expected_gmorans)
    np.testing.assert_almost_equal(interaction['morans_pvals'].values, 3.4125671e-07)

def test_cosine_permutation():
    adata = generate_toy_spatial()
    lr_bivar(adata,
             local_name='cosine',
             global_name=['morans', 'lee'],
             n_perms=100,
             use_raw=True)
    lrdata = adata.obsm['local_scores']

    assert 'pvals' in lrdata.layers.keys()

    np.testing.assert_almost_equal(lrdata[:,'MIF^CD74_CXCR4'].X.mean(), 0.32514292, decimal=6)
    np.testing.assert_almost_equal(np.mean(lrdata[:,'MIF^CD74_CXCR4'].layers['pvals']), 0.601228, decimal=4)

    interaction = lrdata.var[lrdata.var.index == 'S100A9^ITGB2']
    np.testing.assert_almost_equal(interaction['mean'].values, 0.56016606)
    np.testing.assert_almost_equal(interaction['std'].values, 0.33243373)
    np.testing.assert_almost_equal(interaction['morans'].values, expected_gmorans)
    np.testing.assert_almost_equal(interaction['morans_pvals'].values, 0.85)
    np.testing.assert_almost_equal(interaction['lee'].values, expected_glee)
    np.testing.assert_almost_equal(interaction['lee_pvals'].values, 0.93)


def test_morans_pval_none_cats():
    adata = generate_toy_spatial()
    lr_bivar(adata,
             local_name='morans',
             global_name='lee',
             n_perms=None,
             use_raw=True,
             add_categories=True)

    assert 'local_scores' in adata.obsm_keys()
    assert adata.obsm['local_scores'].var.shape == (32, 10)
    lrdata = adata.obsm['local_scores']

    assert 'cats' in lrdata.layers.keys()
    assert lrdata.layers['cats'].sum() == -6197
    assert 'pvals' not in adata.layers.keys()
    interaction = lrdata.var[lrdata.var.index == 'S100A9^ITGB2']
    np.testing.assert_almost_equal(interaction['lee'].values, expected_glee)

def test_wrong_interactions():
    from pytest import raises
    with raises(ValueError):
        lr_bivar(adata,
                 resource_name='mouseconsensus',
                 local_name='morans',
                 n_perms=None,
                 use_raw=True,
                 add_categories=True)
