import unittest
from scanpy.datasets import pbmc68k_reduced
import numpy as np

from liana.utils.pre import check_if_covered, prep_check_adata, format_vars

adata = pbmc68k_reduced()


class TestPre(unittest.TestCase):
    def test_prep_check_adata(self):
        desired = np.array([1.591, 1.591, 1.591, 2.177, 2.544, 1.591, 2.177, 1.591, 2.812, 1.591])
        actual = prep_check_adata(adata, True, None).X.data[0:10]
        np.testing.assert_almost_equal(actual, desired, decimal=3)

    @unittest.expectedFailure
    def test_check_if_covered(self):
        self.assertRaises(check_if_covered(['NOT', 'HERE'], adata.var_names, verbose=True))

    def test_format_vars(self):
        a = ['CD4B_', 'CD8A', 'IL6']
        self.assertFalse(np.array_equal(a, format_vars(a)))


if __name__ == '__main__':
    unittest.main()
