from mudata import MuData
from numpy.testing import assert_almost_equal
from scipy.sparse import csr_matrix
from tests._helpers import get_x

from liana._core._types import MatrixLike
from liana.multisample import mdata_to_anndata
from liana.preprocessing import neg_to_zero, zi_minmax


def test_m_to_adata(toy_mdata: MuData) -> None:
    adata = mdata_to_anndata(
        toy_mdata, x_mod="adata_x", y_mod="adata_y", x_transform=None, y_transform=None, verbose=True
    )
    assert adata.shape == toy_mdata.shape


def test_mdata_transformations(toy_mdata: MuData) -> None:
    # test minmax
    adata = mdata_to_anndata(
        toy_mdata, x_mod="adata_x", y_mod="adata_y", x_transform=zi_minmax, y_transform=zi_minmax, verbose=False
    )
    assert get_x(adata).max() == 1
    assert_almost_equal(get_x(adata).sum(), 1497.3386, decimal=3)

    # test cutoff
    def zi_minmax_cutoff(x: MatrixLike) -> csr_matrix:
        return zi_minmax(x, cutoff=0.25)

    adata = mdata_to_anndata(
        toy_mdata,
        x_mod="adata_x",
        y_mod="adata_y",
        x_transform=zi_minmax_cutoff,
        y_transform=zi_minmax_cutoff,
        verbose=False,
    )
    assert_almost_equal(get_x(adata).sum(), 2120.704, decimal=3)

    # test non-negative
    from scanpy.preprocessing import scale

    scale(toy_mdata.mod["adata_x"])

    adata = mdata_to_anndata(
        toy_mdata, x_mod="adata_x", y_mod="adata_y", x_transform=neg_to_zero, y_transform=None, verbose=False
    )
    assert_almost_equal(get_x(adata).max(), 7.760507, decimal=5)
    assert_almost_equal(get_x(adata).min(), 0, decimal=5)
