import numpy as np
import pytest

from liana._core._pipe_utils._pre import assert_covered, prep_check_adata


def test_prep_check_adata(pbmc68k):
    temp = prep_check_adata(adata=pbmc68k, groupby='bulk_labels', min_cells=0,
                            use_raw=True, layer=None)
    np.testing.assert_almost_equal(np.sum(temp.X.data), 319044.22, decimal=1)

    desired = np.array([2.177, 2.177, 2.544, 2.544, 1.591, 1.591, 1.591, 1.591, 1.591, 1.591])
    np.testing.assert_almost_equal(temp.X.data[0:10], desired, decimal=3)

    # test filtering
    filt = prep_check_adata(adata=pbmc68k, groupby='bulk_labels',
                            min_cells=20, use_raw=True)
    assert len(filt.obs['@label']) == 660


def test_check_if_covered(pbmc68k):
    with pytest.raises(ValueError):
        assert_covered(['NOT', 'HERE'], pbmc68k.var_names, verbose=True)


def test_choose_mtx(pbmc68k):
    # check if default is used correctly
    raw_adata = prep_check_adata(adata=pbmc68k, groupby='bulk_labels', min_cells=5)
    assert np.min(raw_adata.X.data) < 0

    # check if correct layer is returned
    by_layer = prep_check_adata(adata=pbmc68k, groupby='bulk_labels',
                                min_cells=5, use_raw=True)
    by_layer.layers['scaled'] = by_layer.X
    extracted = prep_check_adata(by_layer, groupby='bulk_labels',
                                 min_cells=5, layer='scaled')

    np.testing.assert_almost_equal(by_layer.layers['scaled'].data,
                                   extracted.X.data)


def test_choose_mtx_failure(pbmc68k):
    pbmc68k.layers['scaled_counts'] = pbmc68k.X
    # check exception if both layer and use_raw are provided
    with pytest.raises(ValueError):
        prep_check_adata(adata=pbmc68k, groupby='bulk_labels', min_cells=5,
                         layer='scaled_counts', use_raw=True)

    # check exception if .raw is not initialized
    pbmc68k.raw = None
    with pytest.raises(ValueError):
        prep_check_adata(adata=pbmc68k, groupby='bulk_labels', min_cells=5, use_raw=True)
