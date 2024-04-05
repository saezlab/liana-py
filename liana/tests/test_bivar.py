import numpy as np
from itertools import product

from liana.testing._sample_anndata import generate_toy_mdata
from liana.method.sp._bivariate._SpatialBivariate import bivar

mdata = generate_toy_mdata()
interactions = list(product(mdata.mod['adata_x'].var.index,
                            mdata.mod['adata_y'].var.index))
ones = np.ones((mdata.shape[0], mdata.shape[0]), dtype=np.float64)
mdata.obsp['ones'] = ones


def test_bivar_morans_perms():
    bivar(mdata,
          x_mod='adata_x',
          y_mod='adata_y',
          local_name='morans',
          n_perms=2,
          x_use_raw=False,
          y_use_raw=False,
          interactions=interactions)


    assert 'local_scores' in mdata.mod.keys()
    local_pvals = mdata.mod['local_scores'].layers['pvals']
    np.testing.assert_almost_equal(mdata.mod['local_scores'].X.sum(), -346.55872, decimal=4)
    np.testing.assert_almost_equal(np.mean(local_pvals), 0.52787581, decimal=6)


def test_bivar_nondefault():
    global_stats, local_scores = \
          bivar(mdata,
                x_mod='adata_x',
                y_mod='adata_y',
                local_name='morans',
                global_name=['morans', 'lee'],
                n_perms=0,
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
    np.testing.assert_almost_equal(global_stats['morans_pvals'].mean(), 1, decimal=5)

    local_scores.shape == (700, 100)
    np.testing.assert_almost_equal(np.min(np.min(local_scores.layers['pvals'])), 0.5, decimal=2)


def test_bivar_adata():
      mdata.mod['adata_x'].obsp = mdata.obsp

      bivar(mdata=mdata.mod['adata_x'],
            x_mod=None,
            y_mod=None,
            x_use_raw=False,
            y_use_raw=False,
            local_name='morans',
            connectivity_key='ones',
            interactions=interactions)

      bdata = mdata.mod['adata_x']
      assert 'spatial' in bdata.obsm.keys()
      assert 'louvain' in bdata.uns.keys()
      assert 'spatial_connectivities' in bdata.obsp.keys()


def test_masked_spearman():
    bivar(mdata,
          x_mod='adata_x',
          y_mod='adata_y',
          x_use_raw=False,
          y_use_raw=False,
          local_name='masked_spearman',
          interactions=interactions,
          connectivity_key='ones'
          )

    # check local
    assert 'local_scores' in mdata.mod.keys()
    np.testing.assert_almost_equal(mdata.mod['local_scores'].X.mean(), 0.18438724, decimal=5)

    # check global
    assert mdata.mod['local_scores'].var.shape == (90, 8)
    global_res = mdata.mod['local_scores'].var
    assert set(['mean','std']).issubset(global_res.columns)
    np.testing.assert_almost_equal(global_res['mean'].mean(), 0.18438746, decimal=5)
    np.testing.assert_almost_equal(global_res['std'].mean(), 8.498836e-07, decimal=5)


def test_vectorized_pearson():
    bivar(mdata,
          x_mod='adata_x',
          y_mod='adata_y',
          x_use_raw=False,
          y_use_raw=False,
          local_name='pearson',
          n_perms=100,
          interactions=interactions
          )

    # check local
    assert 'local_scores' in mdata.mod.keys()
    adata = mdata.mod['local_scores']
    np.testing.assert_almost_equal(adata.X.mean(), 0.0011550441, decimal=5)
    np.testing.assert_almost_equal(adata.layers['pvals'].mean(), 0.72182712, decimal=3)

    # check global
    assert mdata.mod['local_scores'].var.shape == (90, 8)
    global_res = mdata.mod['local_scores'].var
    assert set(['mean','std']).issubset(global_res.columns)
    np.testing.assert_almost_equal(global_res['mean'].mean(), 0.0011550438183169959, decimal=5)
    np.testing.assert_almost_equal(global_res['std'].mean(), 0.3227660823917939, decimal=5)
