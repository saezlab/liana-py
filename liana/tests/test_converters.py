from liana.testing._sample_anndata import generate_toy_mdata
from liana.funcomics import mdata_to_anndata
from numpy.testing import assert_almost_equal

def test_m_to_adata():
    mdata = generate_toy_mdata()
    adata = mdata_to_anndata(mdata, x_mod='adata_x', y_mod='adata_y', 
                             x_transform=None, y_transform=None, verbose=False)
    assert adata.shape == mdata.shape
    assert adata.X.max() == 1
    
    # Test with minmax transform
    adata = mdata_to_anndata(mdata, x_mod='adata_x', y_mod='adata_y',
                             x_transform=False, y_transform=False, verbose=True)
    assert_almost_equal(adata.X.max(), 3.431)

