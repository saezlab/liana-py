import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from liana.utils.interpolate_adata import interpolate_adata


def create_test_adata(n_cells, n_genes, spatial_key='spatial'):
    """
    Helper function to create a test AnnData object.
    """
    X = csr_matrix(np.random.rand(n_cells, n_genes))
    obsm = {spatial_key: np.random.rand(n_cells, 2)}
    adata = AnnData(X, obsm=obsm)
    adata.layers['some_layer'] = X
    return adata

@pytest.fixture
def reference_adata():
    return create_test_adata(100, 10)

@pytest.fixture
def target_adata():
    return create_test_adata(80, 10)

def test_basic_interpolation(reference_adata, target_adata):
    result = interpolate_adata(reference=reference_adata, target=target_adata, spatial_key='spatial', use_raw=False)
    assert isinstance(result, AnnData)
    assert result.shape == (100, 10)

@pytest.mark.parametrize("method", ['linear', 'nearest'])
def test_different_methods(reference_adata, target_adata, method):
    result = interpolate_adata(reference=reference_adata, target=target_adata, spatial_key='spatial', method=method, use_raw=False)
    assert isinstance(result, AnnData)

def test_fill_value(reference_adata, target_adata):
    fill_value = -1
    result = interpolate_adata(reference=reference_adata, target=target_adata, spatial_key='spatial', fill_value=fill_value, use_raw=False)
    assert (result.X.data == fill_value).sum() > 0

def test_invalid_spatial_key(reference_adata, target_adata):
    with pytest.raises(KeyError):
        interpolate_adata(reference=reference_adata, target=target_adata, spatial_key='invalid_key')

def test_use_raw_layer_parameters(reference_adata, target_adata):
    result_layer = interpolate_adata(reference=reference_adata, target=target_adata, spatial_key='spatial', layer='some_layer', use_raw=False)
    assert isinstance(result_layer, AnnData)

    with pytest.raises(ValueError):
        interpolate_adata(reference=reference_adata, target=target_adata, spatial_key='spatial', layer='some_layer', use_raw=True)
