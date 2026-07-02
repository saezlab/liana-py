import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from liana.multi import adata_to_views, filter_view_markers, lrdata_to_mudata, lrs_to_views, to_tensor_c2c
from liana.testing import sample_lrs
from liana.testing._sample_anndata import generate_toy_adata
from liana.utils._getters import get_factor_scores, get_variable_loadings

adata = generate_toy_adata()


def _generate_toy_lrdata(n_obs=20, n_lrs=15):
    """Build a small AnnData mimicking `liana.method.inflow`'s output convention."""
    celltypes = ['T_cell', 'B_cell']
    interactions = [f'lig{i}^rec{i}' for i in range(n_lrs)]
    var_names = [f'{ct}^{lr}' for ct in celltypes for lr in interactions]
    rng = np.random.default_rng(42)
    X = rng.random((n_obs, len(var_names)))
    obs = pd.DataFrame(
        {'cell_type': ['T_cell'] * (n_obs // 2) + ['B_cell'] * (n_obs // 2),
         'region': ['A'] * n_obs},
        index=[f'cell{i}' for i in range(n_obs)]
    )
    var = pd.DataFrame(index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def test_to_tensor_c2c():
    """Test to_tensor_c2c."""
    import cell2cell as c2c
    liana_res = sample_lrs(by_sample=True)

    liana_dict = to_tensor_c2c(liana_res=liana_res,
                               sample_key='sample',
                               score_key='specificity_rank',
                               return_dict=True
                               )
    assert isinstance(liana_dict, dict)

    tensor = to_tensor_c2c(liana_res=liana_res,
                           sample_key='sample',
                           score_key='specificity_rank')
    assert isinstance(tensor, c2c.tensor.tensor.PreBuiltTensor)
    assert tensor.sparsity_fraction()==0.0


def test_lrs_to_views():
    """Test lrs_to_views."""
    liana_res = sample_lrs(by_sample=True)
    adata.uns['liana_results'] = liana_res

    mdata = lrs_to_views(adata=adata,
                         sample_key='sample',
                         score_key='specificity_rank',
                         uns_key = 'liana_results',
                         obs_keys = ['case'],
                         source_key='source',
                         target_key='target',
                         ligand_key='ligand_complex',
                         receptor_key='receptor_complex',
                         lr_prop=0.1,
                         lrs_per_sample=0,
                         lrs_per_view=5,
                         samples_per_view=0,
                         min_variance=-1, # don't filter
                         verbose=True
                         )

    assert mdata.shape == (4, 16)
    assert 'case' in mdata.obs.columns
    assert len(mdata.varm_keys())==3


def test_lrs_to_views_batch():
    adata = generate_toy_adata()
    adata.obs['batch'] = 1
    adata2 = adata.copy()
    adata2.obs['batch'] = 2
    adata2.obs['sample'] = adata2.obs['sample'].apply(lambda x: x+'2')
    adata3 = adata.copy()
    adata3.obs['sample'] = adata3.obs['sample'].apply(lambda x: x+'3')
    adata = adata.concatenate([adata2, adata3], join='inner', batch_key='sample_number')

    liana_res = sample_lrs(by_sample=True)
    liana_res2 = liana_res.copy()
    liana_res2['sample'] = liana_res['sample'].apply(lambda x: x+'2')
    liana_res['batch']=1
    liana_res2['batch']=2
    liana_res3 = liana_res.copy()
    liana_res3['sample'] = liana_res3['sample'].apply(lambda x: x+'3')
    # add some variance
    liana_res2['specificity_rank'] = liana_res2['specificity_rank'] + 0.1
    liana_res3['specificity_rank'] = liana_res3['specificity_rank'] + 0.2
    liana_res = pd.concat([liana_res, liana_res2, liana_res3])
    adata.uns['liana_results'] = liana_res

    mdata = lrs_to_views(adata=adata,
                         sample_key='sample',
                         score_key='specificity_rank',
                         uns_key = 'liana_results',
                         obs_keys = ['case', 'batch'],
                         source_key='source',
                         target_key='target',
                         ligand_key='ligand_complex',
                         receptor_key='receptor_complex',
                         lr_prop=0.1,
                         lrs_per_sample=1,
                         lrs_per_view=5,
                         samples_per_view=0,
                         min_variance=0,
                         batch_key='batch',
                         min_var_nbatches=1,
                         verbose=True
                         )

    assert mdata.shape == (12, 16)
    assert 'case' in mdata.obs.columns
    assert 'batch' in mdata.obs.columns
    assert len(mdata.varm_keys())==3

def test_adata_to_views():
    """Test adata_to_views."""
    mdata = adata_to_views(adata,
                           groupby='bulk_labels',
                           sample_key='sample',
                           obs_keys=None,
                           keep_stats=False,
                           verbose=True,
                           psbulk_kwargs={'raw': True,
                                          'skip_checks': True},
                           filter_samples_kwargs={
                               'min_cells': 5,
                               'min_counts': 10,
                           },
                           filter_by_expr_kwargs={
                                 'min_count': 0,
                                 'min_prop': 0,
                                 'min_total_count':0,
                                 'large_n': 0,
                           }
                           )

    assert len(mdata.varm_keys())==9
    assert 'case' not in mdata.obs.columns
    assert mdata.shape == (4, 6885)
    assert 'psbulk_stats' not in mdata.uns.keys()


def test_filter_view_markers():
    mdata = adata_to_views(adata,
                           groupby='bulk_labels',
                           sample_key='sample',
                           obs_keys = ['case'],
                           verbose=True,
                           psbulk_kwargs={'raw': True,
                                          'skip_checks': True},
                           filter_samples_kwargs={
                               'min_cells': 5,
                               'min_counts': 100,
                           },
                           filter_by_expr_kwargs={
                                 'min_count': 100,
                                 'min_prop': 0.1,
                                 'min_total_count':0,
                                 'large_n': 0,
                           }
                           )

    rng = np.random.default_rng(42)
    markers = {}
    for cell_type in mdata.mod.keys():
        markers[cell_type] = rng.choice(adata.var_names, 10).tolist()

    filter_view_markers(mdata, markers, inplace=True)
    assert mdata.mod['Dendritic'].var['highly_variable'].sum() == 33

    filter_view_markers(mdata, markers, var_column=None, inplace=True)
    assert mdata.shape == (4, 74)


def test_lrdata_to_mudata():
    """Test lrdata_to_mudata."""
    lrdata = _generate_toy_lrdata()

    mdata = lrdata_to_mudata(lrdata, min_cells=None, min_features=10,
                             obs_keys=['cell_type', 'region'], verbose=True)

    assert set(mdata.mod.keys()) == {'T_cell', 'B_cell'}
    assert mdata.shape == (20, 30)
    assert mdata['T_cell'].shape == (20, 15)
    assert list(mdata.obs.columns) == ['cell_type', 'region']


def test_lrdata_to_mudata_min_features_drops_modality():
    """A modality with fewer than `min_features` interactions should be dropped, not error."""
    lrdata = _generate_toy_lrdata(n_lrs=15)
    # keep only a handful of B_cell features so it falls below the min_features threshold
    keep = lrdata.var_names[lrdata.var_names.str.startswith('T_cell')].tolist() + \
        lrdata.var_names[lrdata.var_names.str.startswith('B_cell')][:5].tolist()
    lrdata = lrdata[:, keep].copy()

    mdata = lrdata_to_mudata(lrdata, min_cells=None, min_features=10)

    assert set(mdata.mod.keys()) == {'T_cell'}


def test_lrdata_to_mudata_errors():
    lrdata = _generate_toy_lrdata()

    with pytest.raises(TypeError):
        lrdata_to_mudata('not an AnnData')

    with pytest.raises(ValueError):
        lrdata_to_mudata(lrdata, obs_keys=['nonexistent'])

    with pytest.raises(ValueError):
        # no modality can meet an impossibly high min_features
        lrdata_to_mudata(lrdata, min_features=1000)


def test_get_funs():
    liana_res = sample_lrs(by_sample=True)
    adata.uns['liana_results'] = liana_res

    mdata = lrs_to_views(adata=adata,
                         sample_key='sample',
                         score_key='specificity_rank',
                         uns_key = 'liana_results',
                         lr_prop=0.1,
                         lrs_per_sample=0,
                         lrs_per_view=5,
                         samples_per_view=0,
                         min_variance=-1, # don't filter
                         verbose=True
                         )

    # generate random loadings
    mdata.varm['LFs'] = np.random.rand(mdata.shape[1], 5)

    loadings = get_variable_loadings(mdata,
                                     varm_key='LFs',
                                     view_sep=':',
                                     variable_sep='^',
                                     pair_sep='&')
    assert isinstance(loadings, pd.DataFrame)
    assert loadings.shape == (16, 9)

    # dont drop columns & and don't separate
    loadings = get_variable_loadings(mdata,
                                     varm_key='LFs',
                                     drop_columns=False)
    assert isinstance(loadings, pd.DataFrame)
    assert loadings.shape == (16, 6)

    # generate random factor scores
    mdata.obsm['X_mofa'] = np.random.rand(mdata.shape[0], 5)

    scores = get_factor_scores(mdata, obsm_key='X_mofa')
    assert isinstance(scores, pd.DataFrame)
    assert scores.shape == (4, 6)
