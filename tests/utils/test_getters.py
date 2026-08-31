import numpy as np
import pandas as pd
import pytest

from liana.multisample import lrs_to_views
from liana.multisample._getters import get_factor_scores, get_variable_loadings


@pytest.fixture
def mdata(toy_adata, liana_res_by_sample):
    """A MuData of ligand-receptor views, as `lrs_to_views` returns it."""
    toy_adata.uns['liana_results'] = liana_res_by_sample

    return lrs_to_views(adata=toy_adata,
                        sample_key='sample',
                        score_key='specificity_rank',
                        uns_key='liana_results',
                        lr_prop=0.1,
                        lrs_per_sample=0,
                        lrs_per_view=5,
                        samples_per_view=0,
                        min_variance=-1,  # don't filter
                        verbose=True
                        )


def test_get_funs(mdata):
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


def test_get_variable_loadings_from_loadings():
    # MOFA-Flex-style weights: dict of per-view features-by-factors DataFrames,
    # feature names are `sender^ligand^receptor` (no target), factors named "Factor N"
    feats_a = ["astrocyte^Agt^Adra2a", "astrocyte^Fgf1^Egfr"]
    feats_b = ["tanycyte^Efna5^Ephb1", "tanycyte^Rspo3^Lgr6"]
    cols = ["Factor 1", "Factor 2", "Factor 3"]
    weights = {
        "astrocyte": pd.DataFrame(np.arange(6).reshape(2, 3), index=feats_a, columns=cols),
        "tanycyte": pd.DataFrame(np.arange(6, 12).reshape(2, 3), index=feats_b, columns=cols),
    }

    loadings = get_variable_loadings(
        loadings=weights,
        variable_sep="^",
        var_names=["source", "ligand_complex", "receptor_complex"],
    )
    assert isinstance(loadings, pd.DataFrame)
    # 4 features x (3 split cols + 3 factors)
    assert loadings.shape == (4, 6)
    # factor column names are preserved (not renamed to Factor1...)
    assert list(loadings.columns) == ["source", "ligand_complex", "receptor_complex", *cols]
    # sorted by |first factor|, descending
    assert loadings["Factor 1"].abs().is_monotonic_decreasing
    # a concatenated DataFrame gives the same result as the dict input
    from_df = get_variable_loadings(
        loadings=pd.concat(weights.values()),
        variable_sep="^",
        var_names=["source", "ligand_complex", "receptor_complex"],
    )
    assert from_df.equals(loadings)
