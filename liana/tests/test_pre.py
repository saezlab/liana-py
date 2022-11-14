import pytest
from scanpy.datasets import pbmc68k_reduced
import numpy as np

from liana.method._pipe_utils._pre import assert_covered, prep_check_adata, format_vars

adata = pbmc68k_reduced()


def test_prep_check_adata():
    desired = np.array([1.591, 1.591, 1.591, 2.177, 2.544, 1.591, 2.177, 1.591, 2.812, 1.591])
    actual = prep_check_adata(adata, 'bulk_labels', 0, True, None).X.data[0:10]
    np.testing.assert_almost_equal(actual, desired, decimal=3)

    # test filtering
    filt = prep_check_adata(adata=adata, groupby='bulk_labels',
                            min_cells=20, use_raw=True)
    assert len(filt.obs['label']) == 660


def test_check_if_covered():
    with pytest.raises(ValueError):
        assert_covered(['NOT', 'HERE'], adata.var_names, verbose=True)


def test_format_vars():
    a = ['CD4B_', 'CD8A', 'IL6']
    assert (np.array_equal(['CD4B', 'CD8A', 'IL6'], format_vars(a)))


def test_choose_mtx():
    # check if default is used correctly
    raw_adata = prep_check_adata(adata=adata, groupby='bulk_labels', min_cells=5)
    assert np.min(raw_adata.X.data) < 0

    # check if correct layer is returned
    by_layer = prep_check_adata(adata=adata, groupby='bulk_labels',
                                min_cells=5, use_raw=True)
    by_layer.layers['scaled'] = by_layer.X
    extracted = prep_check_adata(by_layer, groupby='bulk_labels',
                                 min_cells=5, layer='scaled')

    np.testing.assert_almost_equal(by_layer.layers['scaled'].data,
                                   extracted.X.data)
