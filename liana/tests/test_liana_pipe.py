from liana.steady.liana_pipe import liana_pipe

import unittest
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
complex_policy = 'min'
key_cols = ['source', 'target', 'ligand_complex', 'receptor_complex']
verbose = False
base = 2.718281828459045
supp_cols = None
resource = None
use_raw = True
layer = None
_return_subunits = False
n_perms = 5
seed = 1337


class TestLianaPipeDefaults(unittest.TestCase):
    # Test ALL Default parameters
    def test_liana_pipe_defaults_shape(self):
        all_defaults = liana_pipe(adata=adata,
                                  groupby=groupby,
                                  resource_name=resource_name,
                                  expr_prop=expr_prop,
                                  de_method=de_method,
                                  base=base,
                                  n_perms=n_perms,
                                  seed=seed,
                                  verbose=verbose,
                                  _key_cols=key_cols,
                                  supp_cols=supp_cols,
                                  resource=resource,
                                  use_raw=use_raw,
                                  layer=layer,
                                  _return_subunits=_return_subunits
                                  )

        self.assertTrue(1271 == all_defaults.shape[0])
        self.assertTrue(18 == all_defaults.shape[1])
        self.assertIn('prop_min', all_defaults.columns)

        exp_defaults = read_csv(test_path.joinpath("data/all_defaults.csv"))
        exp_defaults.index = all_defaults.index
        assert_frame_equal(all_defaults, exp_defaults, check_dtype=False, check_index_type=False)


class TestLianaPipeNotDefaults(unittest.TestCase):
    # Test NOT Default parameters
    def test_liana_pipe_not_defaults(self):
        not_defaults = liana_pipe(adata=adata,
                                  groupby=groupby,
                                  resource_name=resource_name,
                                  expr_prop=0,
                                  de_method='wilcoxon',
                                  base=base,
                                  n_perms=n_perms,
                                  seed=seed,
                                  verbose=verbose,
                                  _key_cols=key_cols,
                                  supp_cols=['ligand_pvals', 'receptor_pvals'],
                                  resource=resource,
                                  use_raw=use_raw,
                                  layer=layer,
                                  _return_subunits=True
                                  )

        self.assertTrue(4400 == not_defaults.shape[0])
        self.assertTrue(19 == not_defaults.shape[1])
        self.assertTrue(all(np.isin(['ligand_pvals', 'receptor_pvals'], not_defaults.columns)))

        exp_defaults = read_csv(test_path.joinpath("data/not_defaults.csv"))
        exp_defaults.index = not_defaults.index
        assert_frame_equal(not_defaults, exp_defaults, check_dtype=False, check_index_type=False)


if __name__ == '__main__':
    unittest.main()
