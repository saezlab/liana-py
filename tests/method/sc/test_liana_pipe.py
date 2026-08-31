from itertools import product

import numpy as np
import pytest
from pandas import DataFrame, read_csv
from pandas.testing import assert_frame_equal

from liana._core._constants import DefaultValues as V
from liana.method.sc._liana_pipe import _calc_log2fc, _expm1_base, liana_pipe

groupby = 'bulk_labels'


@pytest.fixture
def pbmc68k():
    # pbmc68k_reduced ships scaled data in .X and log-norm in .raw; put log-norm in
    # .X so the default (use_raw=False) path is exercised against the reference output.
    from scanpy.datasets import pbmc68k_reduced

    adata = pbmc68k_reduced()
    adata.X = adata.raw.X.copy()
    return adata


# Test ALL Default parameters
def test_liana_pipe_defaults(pbmc68k, data_dir):
    all_defaults = liana_pipe(adata=pbmc68k,
                              groupby=groupby,
                              resource_name=V.resource_name,
                              groupby_pairs=V.groupby_pairs,
                              expr_prop=V.expr_prop,
                              min_cells=V.min_cells,
                              de_method=V.de_method,
                              base=V.logbase,
                              n_perms=V.n_perms,
                              seed=V.seed,
                              verbose=V.seed,
                              supp_columns=[],
                              resource=V.resource,
                              use_raw=False,  # local fixture puts log-norm in .X
                              layer=V.layer,
                              n_jobs=1,
                              interactions=V.interactions,
                              )

    assert 'prop_min' in all_defaults.columns
    all_defaults = all_defaults.sort_values(by=list(all_defaults.columns))

    exp_defaults = read_csv(data_dir / "all_defaults.csv", index_col=0)
    exp_defaults = exp_defaults.sort_values(by=list(all_defaults.columns))
    exp_defaults.index = all_defaults.index

    assert_frame_equal(all_defaults, exp_defaults, check_dtype=False,
                       check_exact=False, check_index_type=False, rtol=1e-3)


# Test NOT Default parameters
def test_liana_pipe_not_defaults(pbmc68k, data_dir):
    not_defaults = liana_pipe(adata=pbmc68k,
                              groupby=groupby,
                              resource_name=V.resource_name,
                              expr_prop=0.2,
                              groupby_pairs=V.groupby_pairs,
                              min_cells=V.min_cells,
                              de_method='wilcoxon',
                              base=V.logbase,
                              n_perms=V.n_perms,
                              seed=V.seed,
                              verbose=V.verbose,
                              supp_columns=['ligand_pvals', 'receptor_pvals'],
                              resource=V.resource,
                              use_raw=False,  # local fixture puts log-norm in .X
                              layer=V.layer,
                              return_all_lrs=True,
                              n_jobs=1,
                              interactions=V.interactions,
                              )

    assert all(np.isin(['lrs_to_keep'], not_defaults.columns))
    assert all(np.isin(['ligand_pvals', 'receptor_pvals'], not_defaults.columns))
    not_defaults = not_defaults.sort_values(list(not_defaults.columns))

    exp_defaults = read_csv(data_dir / "not_defaults.csv", index_col=0)
    exp_defaults = exp_defaults.sort_values(list(not_defaults.columns))
    exp_defaults.index = not_defaults.index
    assert_frame_equal(not_defaults, exp_defaults, check_dtype=False,
                       check_index_type=False, check_exact=False, rtol=1e-3)



def test_liana_pipe_subset(pbmc68k):
    cts = ['CD34+', 'Dendritic', 'CD56 NK', 'CD19+ B']
    groupby_pairs = list(product(cts, cts))
    groupby_pairs = DataFrame(groupby_pairs, columns=['source', 'target'])
    groupby_pairs = groupby_pairs[groupby_pairs['source'] == 'Dendritic']

    subset = liana_pipe(adata=pbmc68k,
                        groupby=groupby,
                        resource_name=V.resource_name,
                        expr_prop=0.05,
                        groupby_pairs=groupby_pairs,
                        min_cells=V.min_cells,
                        de_method=V.de_method,
                        base=V.logbase,
                        n_perms=V.n_perms,
                        seed=V.seed,
                        verbose=V.verbose,
                        resource=V.resource,
                        use_raw=False,  # local fixture puts log-norm in .X
                        layer=V.layer,
                        n_jobs=1,
                        interactions=V.interactions,
                        )

    assert subset.shape == (46, 23)


def test_expm1_fun(pbmc68k):
    expm1_mat = _expm1_base(V.logbase, pbmc68k.raw.X.data)
    np.testing.assert_almost_equal(np.sum(expm1_mat), 1057526.4, decimal=1)


def test_calc_log2fc(pbmc68k):
    pbmc68k.layers['normcounts'] = pbmc68k.raw.X.copy()
    pbmc68k.layers['normcounts'].data = _expm1_base(V.logbase, pbmc68k.raw.X.data)
    pbmc68k.obs['@label'] = pbmc68k.obs.bulk_labels
    np.testing.assert_almost_equal(np.mean(_calc_log2fc(pbmc68k, "Dendritic")), -0.123781264)


def test_calc_log2fc_no_rest_raises(pbmc68k):
    single_label = pbmc68k[pbmc68k.obs.bulk_labels == "Dendritic"].copy()
    single_label.layers['normcounts'] = single_label.raw.X.copy()
    single_label.layers['normcounts'].data = _expm1_base(V.logbase, single_label.raw.X.data)
    single_label.obs['@label'] = single_label.obs.bulk_labels

    with pytest.raises(ValueError, match="Cannot compute log2FC"):
        _calc_log2fc(single_label, "Dendritic")
