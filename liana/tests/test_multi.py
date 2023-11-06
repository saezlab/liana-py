from liana.multi import to_tensor_c2c, adata_to_views, lrs_to_views, filter_view_markers
from liana.utils._getters import get_factor_scores, get_variable_loadings
from liana.testing import sample_lrs

import numpy as np
import pandas as pd

import cell2cell as c2c

from liana.testing._sample_anndata import generate_toy_adata

adata = generate_toy_adata()


def test_to_tensor_c2c():
    """Test to_tensor_c2c."""
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



def test_adata_to_views():
    """Test adata_to_views."""
    mdata = adata_to_views(adata,
                           groupby='bulk_labels',
                           sample_key='sample',
                           obs_keys=None,
                           min_cells=5,
                           min_counts=10,
                           keep_stats=False,
                           mode='sum',
                           verbose=True,
                           use_raw=True,
                           min_smpls=2,
                           # filter features
                           min_count=0,
                           min_total_count=0,
                           large_n=0,
                           min_prop=0,
                           skip_checks=True # skip because it's log-normalized
                           )

    assert len(mdata.varm_keys())==9
    assert 'case' not in mdata.obs.columns
    assert mdata.shape == (4, 6201)
    assert 'psbulk_stats' not in mdata.uns.keys()

    # test feature level filtering (with default values)
    mdata = adata_to_views(adata,
                           groupby='bulk_labels',
                           sample_key='sample',
                           obs_keys = ['case'],
                           mode='sum',
                           keep_stats=True,
                           verbose=True,
                           use_raw=True,
                           skip_checks=True
                           )

    assert len(mdata.varm_keys())==7
    assert 'case' in mdata.obs.columns
    assert mdata.shape == (4, 1598)
    assert mdata.uns['psbulk_stats'] is not None


def test_filter_view_markers():
    mdata = adata_to_views(adata,
                           groupby='bulk_labels',
                           sample_key='sample',
                           obs_keys = ['case'],
                           mode='sum',
                           verbose=True,
                           use_raw=True,
                           skip_checks=True
                           )

    rng = np.random.default_rng(42)
    markers = {}
    for cell_type in mdata.mod.keys():
        markers[cell_type] = rng.choice(adata.var_names, 10).tolist()

    filter_view_markers(mdata, markers, inplace=True)
    assert mdata.mod['Dendritic'].var['highly_variable'].sum() == 139

    filter_view_markers(mdata, markers, var_column=None, inplace=True)
    assert mdata.shape == (4, 1471)


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
