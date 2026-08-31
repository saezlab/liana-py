from numpy.testing import assert_almost_equal

from liana.multisample import mdata_to_anndata
from liana.preprocessing import neg_to_zero, zi_minmax


def test_m_to_adata(toy_mdata):
    adata = mdata_to_anndata(toy_mdata, x_mod='adata_x', y_mod='adata_y',
                            x_transform=False, y_transform=False, verbose=True)
    assert adata.shape == toy_mdata.shape


def test_mdata_transformations(toy_mdata):
    # test minmax
    adata = mdata_to_anndata(toy_mdata, x_mod='adata_x', y_mod='adata_y',
                             x_transform=zi_minmax, y_transform=zi_minmax,
                             verbose=False)
    assert adata.X.max() == 1
    assert_almost_equal(adata.X.sum(), 1497.3386, decimal=3)

    # test cutoff
    def zi_minmax_cutoff(x):
        x = zi_minmax(x, cutoff=0.25)
        return x

    adata = mdata_to_anndata(toy_mdata, x_mod='adata_x', y_mod='adata_y',
                             x_transform=zi_minmax_cutoff, y_transform=zi_minmax_cutoff,
                             verbose=False)
    assert_almost_equal(adata.X.sum(), 2120.704, decimal=3)

    # test non-negative
    from scanpy.pp import scale
    scale(toy_mdata.mod['adata_x'])

    adata = mdata_to_anndata(toy_mdata, x_mod='adata_x', y_mod='adata_y',
                             x_transform=neg_to_zero, y_transform=False,
                             verbose=False)
    assert_almost_equal(adata.X.max(), 7.760507, decimal=5)
    assert_almost_equal(adata.X.min(), 0, decimal=5)
