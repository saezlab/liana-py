from liana.method._liana_pipe import liana_pipe, _expm1_base, _calc_log2fc

import pathlib
from scanpy.datasets import pbmc68k_reduced
from pandas.testing import assert_frame_equal
from pandas import read_csv
import numpy as np

test_path = pathlib.Path(__file__).parent
adata = pbmc68k_reduced()

groupby = 'bulk_labels'
de_method = 't-test'
resource_name = 'consensus'
expr_prop = 0.1
min_cells = 5
complex_policy = 'min'
key_cols = ['source', 'target', 'ligand_complex', 'receptor_complex']
verbose = False
base = 2.718281828459045
resource = None
use_raw = True
layer = None
n_perms = 5
seed = 1337


# Test ALL Default parameters
def test_liana_pipe_defaults_shape():
    all_defaults = liana_pipe(adata=adata,
                              groupby=groupby,
                              resource_name=resource_name,
                              expr_prop=expr_prop,
                              min_cells=min_cells,
                              de_method=de_method,
                              base=base,
                              n_perms=n_perms,
                              seed=seed,
                              verbose=verbose,
                              _key_cols=key_cols,
                              supp_columns=[],
                              resource=resource,
                              use_raw=use_raw,
                              layer=layer
                              )

    assert 1288 == all_defaults.shape[0]
    assert 21 == all_defaults.shape[1]
    assert 'prop_min' in all_defaults.columns

    exp_defaults = read_csv(test_path.joinpath("data/all_defaults.csv"), index_col=0)
    exp_defaults.index = all_defaults.index
    assert_frame_equal(all_defaults, exp_defaults, check_dtype=False,
                       check_exact=False, check_less_precise=True, check_index_type=False)


# Test NOT Default parameters
def test_liana_pipe_not_defaults():
    not_defaults = liana_pipe(adata=adata,
                              groupby=groupby,
                              resource_name=resource_name,
                              expr_prop=0.2,
                              min_cells=min_cells,
                              de_method='wilcoxon',
                              base=base,
                              n_perms=n_perms,
                              seed=seed,
                              verbose=verbose,
                              _key_cols=key_cols,
                              supp_columns=['ligand_pvals', 'receptor_pvals'],
                              resource=resource,
                              use_raw=use_raw,
                              layer=layer,
                              return_all_lrs=True
                              )

    assert 4200 == not_defaults.shape[0]
    assert 24 == not_defaults.shape[1]
    assert all(np.isin(['lrs_to_keep'], not_defaults.columns))
    assert all(np.isin(['ligand_pvals', 'receptor_pvals'], not_defaults.columns))

    exp_defaults = read_csv(test_path.joinpath("data/not_defaults.csv"), index_col=0)
    exp_defaults.index = not_defaults.index
    assert_frame_equal(not_defaults, exp_defaults, check_dtype=False,
                       check_index_type=False, check_exact=False, check_less_precise=True)


def test_expm1_fun():
    expm1_mat = _expm1_base(base, adata.raw.X.data)
    np.testing.assert_almost_equal(np.sum(expm1_mat), 1057526.4, decimal=1)


def test_calc_log2fc():
    adata.layers['normcounts'] = adata.raw.X.copy()
    adata.layers['normcounts'].data = _expm1_base(base, adata.raw.X.data)
    adata.obs['label'] = adata.obs.bulk_labels
    np.testing.assert_almost_equal(np.mean(_calc_log2fc(adata, "Dendritic")), -0.123781264)
