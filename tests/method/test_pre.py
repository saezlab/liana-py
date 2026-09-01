import numpy as np
import pytest
from anndata import AnnData
from tests._helpers import as_frame, get_csr, get_layer_csr, get_x

from liana._core._pipe_utils._pre import assert_covered, prep_check_adata


def test_prep_check_adata(pbmc68k: AnnData) -> None:
    temp = prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=0, use_raw=True, layer=None)
    np.testing.assert_almost_equal(np.sum(get_csr(temp).data), 319044.22, decimal=1)

    desired = np.array([2.177, 2.177, 2.544, 2.544, 1.591, 1.591, 1.591, 1.591, 1.591, 1.591])
    np.testing.assert_almost_equal(get_csr(temp).data[0:10], desired, decimal=3)

    # test filtering
    filt = prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=20, use_raw=True)
    assert len(filt.obs["@label"]) == 660


def test_default_reads_X_not_raw(pbmc68k: AnnData) -> None:
    # Guards the public default: use_raw defaults to False, so methods read .X.
    # pbmc68k_reduced ships scaled data in .X and log-norm in .raw, so the two
    # paths give different results -- the default must match the .X path.
    from liana.method import cellphonedb

    default = as_frame(cellphonedb(pbmc68k, groupby="bulk_labels", n_perms=None, inplace=False))
    from_x = as_frame(cellphonedb(pbmc68k, groupby="bulk_labels", n_perms=None, inplace=False, use_raw=False))
    from_raw = as_frame(cellphonedb(pbmc68k, groupby="bulk_labels", n_perms=None, inplace=False, use_raw=True))

    assert default.equals(from_x)  # default == .X
    assert not default.equals(from_raw)  # and differs from .raw


def test_check_if_covered(pbmc68k: AnnData) -> None:
    with pytest.raises(ValueError):
        assert_covered(["NOT", "HERE"], pbmc68k.var_names, verbose=True)


def test_choose_mtx(pbmc68k: AnnData) -> None:
    # check if default is used correctly
    raw_adata = prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5)
    assert np.min(get_csr(raw_adata).data) < 0

    # check if correct layer is returned
    by_layer = prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5, use_raw=True)
    by_layer.layers["scaled"] = get_x(by_layer)
    extracted = prep_check_adata(by_layer, groupby="bulk_labels", min_cells=5, layer="scaled")

    np.testing.assert_almost_equal(get_layer_csr(by_layer, "scaled").data, get_csr(extracted).data)


def test_choose_mtx_failure(pbmc68k: AnnData) -> None:
    pbmc68k.layers["scaled_counts"] = get_x(pbmc68k)
    # check exception if both layer and use_raw are provided
    with pytest.raises(ValueError):
        prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5, layer="scaled_counts", use_raw=True)

    # check exception if .raw is not initialized
    del pbmc68k.raw
    with pytest.raises(ValueError):
        prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5, use_raw=True)
