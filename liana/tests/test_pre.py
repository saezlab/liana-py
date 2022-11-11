import pytest
from scanpy.datasets import pbmc68k_reduced
import numpy as np

from liana.utils.pre import assert_covered, prep_check_adata, format_vars

adata = pbmc68k_reduced()


def test_prep_check_adata():
    desired = np.array([1.591, 1.591, 1.591, 2.177, 2.544, 1.591, 2.177, 1.591, 2.812, 1.591])
    actual = prep_check_adata(adata, True, None).X.data[0:10]
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_check_if_covered():
    with pytest.raises(ValueError):
        assert_covered(['NOT', 'HERE'], adata.var_names, verbose=True)


def test_format_vars():
    a = ['CD4B_', 'CD8A', 'IL6']
    assert (np.array_equal(['CD4B', 'CD8A', 'IL6'], format_vars(a)))

