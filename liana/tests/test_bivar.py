import numpy as np
from itertools import product

from liana.testing._sample_anndata import generate_toy_mdata, generate_toy_spatial, generate_anndata
from liana.testing._sample_resource import sample_resource
from liana.method.sp._bivariate._spatial_bivariate import bivariate
from liana._constants import DefaultValues as V
from liana.utils.transform import zi_minmax

mdata = generate_toy_mdata()
interactions = list(product(mdata.mod['adata_x'].var.index,
                            mdata.mod['adata_y'].var.index))
ones = np.ones((mdata.shape[0], mdata.shape[0]), dtype=np.float64)
mdata.obsp['ones'] = ones


def test_bivar_morans_perms():
    bivariate(mdata,
              x_mod='adata_x',
              y_mod='adata_y',
              local_name='morans',
              n_perms=2,
              nz_prop=0,
              x_use_raw=False,
              y_use_raw=False,
              interactions=interactions)


    assert 'local_scores' in mdata.mod.keys()
    local_pvals = mdata.mod['local_scores'].layers['pvals']
    np.testing.assert_almost_equal(mdata.mod['local_scores'].X.sum(), -346.55872, decimal=4)
    np.testing.assert_almost_equal(np.mean(local_pvals), 0.52787581, decimal=6)


def test_bivar_nondefault():
    global_stats, local_scores = \
          bivariate(mdata,
                x_mod='adata_x',
                y_mod='adata_y',
                local_name='morans',
                global_name=['morans', 'lee'],
                n_perms=0,
                nz_prop=0,
                connectivity_key='ones',
                remove_self_interactions=False,
                x_layer = "scaled",
                y_layer = "scaled",
                x_use_raw=False,
                y_use_raw=False,
                inplace=False,
                add_categories=True,
                interactions=interactions
                )

    np.testing.assert_almost_equal(global_stats['morans'].sum(), 0)
    np.testing.assert_almost_equal(global_stats['lee'].sum(), 0)
    assert global_stats['lee_pvals'].unique()[0] is None
    assert 'morans_pvals' in global_stats.columns

    local_scores.shape == (700, 100)
    np.testing.assert_almost_equal(np.min(np.min(local_scores.layers['pvals'])), 0.5, decimal=2)


def test_masked_spearman():
    bivariate(mdata,
              x_mod='adata_x',
              y_mod='adata_y',
              x_use_raw=False,
              y_use_raw=False,
              nz_prop=0,
              local_name='masked_spearman',
              interactions=interactions,
              connectivity_key='ones'
              )

    assert 'local_scores' in mdata.mod.keys()
    np.testing.assert_almost_equal(mdata.mod['local_scores'].X.mean(), 0.18438724, decimal=5)

    assert mdata.mod['local_scores'].var.shape == (90, 8)
    global_res = mdata.mod['local_scores'].var
    assert set(['mean','std']).issubset(global_res.columns)
    np.testing.assert_almost_equal(global_res['mean'].mean(), 0.18438746, decimal=5)
    np.testing.assert_almost_equal(global_res['std'].mean(), 8.498836e-07, decimal=5)


def test_vectorized_spearman():
    bivariate(mdata,
              x_mod='adata_x',
              y_mod='adata_y',
              x_use_raw=False,
              y_use_raw=False,
              local_name='spearman',
              nz_prop=0,
              n_perms=2,
              interactions=interactions,
              )
    assert 'local_scores' in mdata.mod.keys()
    bdata = mdata.mod['local_scores']
    np.testing.assert_almost_equal(bdata.X.mean(), 0.0077174882, decimal=5)
    np.testing.assert_almost_equal(bdata.layers['pvals'].mean(), 0.6153921568, decimal=3)

    assert mdata.mod['local_scores'].var.shape == (90, 8)
    global_res = mdata.mod['local_scores'].var
    assert set(['mean','std']).issubset(global_res.columns)
    np.testing.assert_almost_equal(global_res['mean'].mean(), 0.0077174, decimal=5)
    np.testing.assert_almost_equal(global_res['std'].mean(), 0.46906388, decimal=5)

### Test on AnnData and LRs
 # NOTE: these should be the same regardless of the local function
adata = generate_toy_spatial()
expected_gmorans = 0.0994394
expected_glee = 0.04854206

def test_morans_analytical():
    bivariate(adata,
              local_name='morans',
              global_name=['morans'],
              resource_name=V.resource_name,
              n_perms=0,
              use_raw=True,
              mask_negatives=True
              )
    assert 'local_scores' in adata.obsm_keys()
    assert 'spatial' in adata.obsm.keys()
    assert 'louvain' in adata.uns.keys()
    assert 'spatial_connectivities' in adata.obsp.keys()

    lrdata = adata.obsm['local_scores']

    assert 'pvals' in lrdata.layers.keys()

    np.testing.assert_almost_equal(np.mean(lrdata[:,'MIF^CD74_CXCR4'].X), 0.12803833, decimal=6)
    np.testing.assert_almost_equal(np.mean(lrdata[:,'MIF^CD74_CXCR4'].layers['pvals']), 0.8764923, decimal=6)

    interaction = lrdata.var[lrdata.var.index == 'S100A9^ITGB2']
    np.testing.assert_almost_equal(interaction['morans'].values, expected_gmorans)
    np.testing.assert_almost_equal(interaction['morans_pvals'].values, 3.4125671e-07)

def test_cosine_permutation():
    adata.layers['array'] = adata.raw.X.A
    bivariate(adata,
              local_name='cosine',
              global_name=['morans', 'lee'],
              resource_name='consensus',
              n_perms=100,
              use_raw=False,
              layer='array'
              )
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


def test_jaccard_pval_none_cats():
    bivariate(adata,
              local_name='jaccard',
              global_name='lee',
              resource_name='consensus',
              n_perms=None,
              use_raw=True,
              add_categories=True
              )

    assert 'local_scores' in adata.obsm_keys()
    assert adata.obsm['local_scores'].var.shape == (32, 10)
    lrdata = adata.obsm['local_scores']

    assert 'cats' in lrdata.layers.keys()
    assert lrdata.layers['cats'].sum() == -6197
    assert 'pvals' not in adata.layers.keys()
    interaction = lrdata.var[lrdata.var.index == 'S100A9^ITGB2']
    np.testing.assert_almost_equal(interaction['lee'].values, expected_glee)

    np.testing.assert_almost_equal(lrdata[:,'S100A9^ITGB2'].X.mean(), 0.4117572, decimal=6)


def test_bivar_product():
    conn = mdata.obsp['spatial_connectivities']
    mdata.obsp['norm'] = conn / conn.sum(axis=1)
    bivariate(mdata,
              x_mod='adata_x',
              y_mod='adata_y',
              x_transform=zi_minmax,
              y_transform=zi_minmax,
              x_use_raw=False,
              y_use_raw=False,
              connectivity_key='norm',
              local_name='product',
              global_name=None,
              interactions=interactions,
              n_perms=None,
              add_categories=True,
              key_added='product'
              )
    assert 'product' in mdata.mod.keys()
    bdata = mdata.mod['product']
    assert 'cats' in bdata.layers.keys()
    assert bdata.uns is not None
    assert bdata.obsm is not None
    assert bdata.obsp is not None
    np.testing.assert_almost_equal(bdata.X.max(), 0.63145)
    assert 'lee' not in bdata.var.columns

def test_large_adata():
    adata = generate_anndata(n_obs=10001)
    resource = sample_resource(adata, n_lrs=20)
    bivariate(adata,
              resource=resource,
              local_name='pearson',
              global_name='morans',
              n_perms=None,
              use_raw=False,
              add_categories=False
              )
    lrdata = adata.obsm['local_scores']
    np.testing.assert_almost_equal(lrdata.X.mean(), 0.00048977, decimal=4)
    np.testing.assert_almost_equal(lrdata.var['morans'].mean(), 0.00030397394, decimal=4)


def test_wrong_interactions():
    from pytest import raises
    with raises(ValueError):
        bivariate(adata,
                 resource_name='mouseconsensus',
                 local_name='morans',
                 n_perms=None,
                 use_raw=True,
                 add_categories=True
                 )

def test_wrong_kwargs():
    from pytest import raises
    with raises(ValueError):
        bivariate(adata,
                 resource_name='mouseconsensus',
                 local_name='morans',
                 n_perms=None,
                 use_raw=True,
                 add_categories=True,
                 life='is good'
                 )


def test_show_bivariate():
    local_scores = bivariate.show_functions()
    assert local_scores.shape == (8, 3)
