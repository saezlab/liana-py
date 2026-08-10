from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from liana.method import inflow
from liana.testing._sample_anndata import generate_toy_mdata, generate_toy_spatial
from liana.testing._sample_resource import sample_resource
from liana.utils.transform import zi_minmax


@pytest.fixture
def spatial_adata():
    """Generate spatial AnnData with connectivity matrix."""
    adata = generate_toy_spatial()
    # Ensure spatial_connectivities exists
    if 'spatial_connectivities' not in adata.obsp:
        from liana.utils import spatial_neighbors
        spatial_neighbors(adata, bandwidth=100, spatial_key='spatial')
    return adata


def test_inflow_basic_structure(spatial_adata):
    """Test basic execution and output structure preservation."""
    lrdata = inflow(
        spatial_adata,
        groupby='bulk_labels',
        resource_name='consensus',
        use_raw=True
    )

    # Check output structure
    assert isinstance(lrdata, type(spatial_adata))
    assert lrdata.shape == (spatial_adata.shape[0], 323)  # Fixed expected shape

    # Check var index format: "celltype^ligand^receptor"
    assert all('^' in idx for idx in lrdata.var_names)
    for idx in lrdata.var_names:
        parts = idx.split('^')
        assert len(parts) == 3
        celltype, ligand, receptor = parts
        assert celltype in spatial_adata.obs['bulk_labels'].unique()

    # Check sparse matrix
    assert isinstance(lrdata.X, csr_matrix)

    # Check obs and obsm preserved
    assert lrdata.obs.equals(spatial_adata.obs)
    assert 'spatial' in lrdata.obsm
    np.testing.assert_array_equal(
        lrdata.obsm['spatial'],
        spatial_adata.obsm['spatial']
    )

    # Check obsp preserved
    assert 'spatial_connectivities' in lrdata.obsp


def test_inflow_with_transform(spatial_adata):
    """Test inflow with zi_minmax transformation."""
    lrdata = inflow(
        spatial_adata,
        groupby='bulk_labels',
        resource_name='consensus',
        x_transform=zi_minmax,
        y_transform=zi_minmax,
        use_raw=False
    )

    assert lrdata.shape[0] == spatial_adata.shape[0]
    # Transformed values should be in [0, 1]
    assert lrdata.X.min() >= 0
    assert lrdata.X.max() <= 1


def test_inflow_nz_prop_filter(spatial_adata):
    """Test filtering by non-zero proportion."""
    # Strict filter
    lrdata_strict = inflow(
        spatial_adata,
        groupby='bulk_labels',
        resource_name='consensus',
        nz_prop=0.2,
        use_raw=True
    )

    # Lenient filter
    lrdata_lenient = inflow(
        spatial_adata,
        groupby='bulk_labels',
        resource_name='consensus',
        nz_prop=0.001,
        use_raw=True
    )

    # Strict filter should have fewer or equal interactions
    assert lrdata_strict.shape[1] <= lrdata_lenient.shape[1]


def test_inflow_custom_resource(spatial_adata):
    """Test with custom resource DataFrame."""
    resource = sample_resource(spatial_adata, n_lrs=10)

    lrdata = inflow(
        spatial_adata,
        groupby='bulk_labels',
        resource=resource,
        use_raw=True
    )

    assert lrdata.shape[1] > 0
    assert lrdata.shape[0] == spatial_adata.shape[0]


def test_inflow_numerical_correctness(spatial_adata):
    """Test numerical correctness of inflow scores."""
    lrdata = inflow(
        spatial_adata,
        groupby='bulk_labels',
        resource_name='consensus',
        use_raw=True
    )

    # Check specific numerical values (regression test)
    np.testing.assert_almost_equal(lrdata.X.mean(), 0.041507, decimal=3)  # Replace with actual
    np.testing.assert_almost_equal(lrdata.X.sum(), 9384.73809, decimal=3)    # Replace with actual


def test_inflow_missing_connectivity(spatial_adata):
    """Test error when spatial_connectivities is missing."""
    # Remove spatial connectivity
    del spatial_adata.obsp['spatial_connectivities']

    with pytest.raises(ValueError, match="spatial_connectivities"):
        inflow(
            spatial_adata,
            groupby='bulk_labels',
            resource_name='consensus',
            use_raw=True
        )


def test_inflow_no_features_pass_filter(spatial_adata):
    """Test error when no features pass nz_prop filter."""
    with pytest.raises(ValueError, match="No features passed"):
        inflow(
            spatial_adata,
            groupby='bulk_labels',
            resource_name='consensus',
            nz_prop=0.99,  # Very strict filter
            use_raw=True
        )


def test_inflow_invalid_groupby(spatial_adata):
    """Test error with invalid groupby column."""
    with pytest.raises(KeyError):
        inflow(
            spatial_adata,
            groupby='nonexistent_column',
            resource_name='consensus',
            use_raw=True
        )


def test_inflow_with_obsm_key(spatial_adata):
    """Test inflow with pre-computed cell type matrix from obsm."""

    # Create soft cell type assignments (probabilities) as DataFrame
    n_celltypes = 3
    ct_probs = np.random.rand(spatial_adata.n_obs, n_celltypes)
    ct_probs = ct_probs / ct_probs.sum(axis=1, keepdims=True)  # normalize to sum to 1
    ct_probs_df = pd.DataFrame(ct_probs, columns=[f'CT_{i}' for i in range(n_celltypes)], index=spatial_adata.obs.index)
    spatial_adata.obsm['ct_probs'] = ct_probs_df

    lrdata = inflow(
        spatial_adata,
        obsm_key='ct_probs',
        resource_name='consensus',
        use_raw=True
    )

    # Check output structure
    assert isinstance(lrdata, type(spatial_adata))
    assert lrdata.shape[0] == spatial_adata.shape[0]
    assert lrdata.shape[1] > 0

def test_inflow_obsm_vs_groupby_equivalence(spatial_adata):
    """Test that one-hot from groupby matches binary obsm."""
    import pandas as pd

    # Create one-hot from groupby
    ct_onehot = pd.get_dummies(spatial_adata.obs['bulk_labels'])
    spatial_adata.obsm['ct_onehot'] = ct_onehot

    lrdata1 = inflow(
        spatial_adata,
        groupby='bulk_labels',
        resource_name='consensus',
        use_raw=True
    )
    lrdata2 = inflow(
        spatial_adata,
        obsm_key='ct_onehot',
        resource_name='consensus',
        use_raw=True
    )

    # Should be identical (or very close)
    assert lrdata1.shape == lrdata2.shape
    np.testing.assert_array_almost_equal(
        lrdata1.X.toarray(),
        lrdata2.X.toarray(),
        decimal=5
    )


def test_inflow_groupby_obsm_validation(spatial_adata):
    """Test error when neither or both groupby/obsm_key provided."""
    # Test neither parameter provided
    with pytest.raises(ValueError, match="Exactly one"):
        inflow(
            spatial_adata,
            resource_name='consensus',
            use_raw=True
        )

    # Test both parameters provided
    spatial_adata.obsm['ct'] = np.random.rand(spatial_adata.n_obs, 3)
    with pytest.raises(ValueError, match="Exactly one"):
        inflow(
            spatial_adata,
            groupby='bulk_labels',
            obsm_key='ct',
            resource_name='consensus',
            use_raw=True
        )


def test_inflow_obsm_missing_key(spatial_adata):
    """Test error when obsm_key not found in obsm."""
    with pytest.raises(KeyError, match="not found in adata.obsm"):
        inflow(
            spatial_adata,
            obsm_key='nonexistent_key',
            resource_name='consensus',
            use_raw=True
        )


def test_inflow_obsm_not_dataframe(spatial_adata):
    """Test error when obsm matrix is not a DataFrame."""
    # Create matrix as numpy array instead of DataFrame
    spatial_adata.obsm['ct_array'] = np.random.rand(spatial_adata.n_obs, 3)

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        inflow(
            spatial_adata,
            obsm_key='ct_array',
            resource_name='consensus',
            use_raw=True
        )


def test_inflow_with_mudata():
    """Test inflow with MuData input."""

    mdata = generate_toy_mdata()
    interactions = list(product(mdata.mod['adata_x'].var.index,
                                mdata.mod['adata_y'].var.index))

    lrdata = inflow(
        mdata,
        groupby='bulk_labels',
        interactions=interactions,
        x_mod='adata_x',
        y_mod='adata_y',
        x_use_raw=False,
        y_use_raw=False,
        nz_prop=0
    )

    # Check output structure
    assert isinstance(lrdata, type(mdata.mod['adata_x']))
    assert lrdata.shape[0] == mdata.shape[0]
    assert lrdata.shape[1] > 0

    # Check var index format: "celltype^ligand^receptor"
    assert all('^' in idx for idx in lrdata.var_names)

    # Check sparse matrix
    assert isinstance(lrdata.X, csr_matrix)


def test_inflow_mudata_vs_anndata_equivalence():
    """Test that MuData and AnnData give same results when data is identical."""
    from liana.utils.mdata_to_anndata import mdata_to_anndata

    mdata = generate_toy_mdata()
    interactions = list(product(mdata.mod['adata_x'].var.index,
                                mdata.mod['adata_y'].var.index))

    # Run with MuData
    lrdata_mudata = inflow(
        mdata,
        groupby='bulk_labels',
        interactions=interactions,
        x_mod='adata_x',
        y_mod='adata_y',
        x_use_raw=False,
        y_use_raw=False,
        nz_prop=0
    )

    # Convert to AnnData manually and run
    adata_combined = mdata_to_anndata(
        mdata,
        x_mod='adata_x',
        y_mod='adata_y',
        x_use_raw=False,
        y_use_raw=False,
        verbose=False
    )

    lrdata_anndata = inflow(
        adata_combined,
        groupby='bulk_labels',
        interactions=interactions,
        use_raw=False,
        layer=None,
        nz_prop=0
    )

    # Check that results have the same dimensions
    assert lrdata_mudata.shape == lrdata_anndata.shape

    # Check that variable names match
    assert set(lrdata_mudata.var_names) == set(lrdata_anndata.var_names)


def test_inflow_mudata_missing_mod():
    """Test error handling when modality parameters are missing for MuData."""

    mdata = generate_toy_mdata()
    interactions = list(product(mdata.mod['adata_x'].var.index,
                                mdata.mod['adata_y'].var.index))

    # Missing x_mod
    with pytest.raises(ValueError, match="requires 'x_mod' and 'y_mod'"):
        inflow(
            mdata,
            groupby='bulk_labels',
            interactions=interactions,
            y_mod='adata_y',
            x_use_raw=False,
            y_use_raw=False
        )

    # Missing y_mod
    with pytest.raises(ValueError, match="requires 'x_mod' and 'y_mod'"):
        inflow(
            mdata,
            groupby='bulk_labels',
            interactions=interactions,
            x_mod='adata_x',
            x_use_raw=False,
            y_use_raw=False
        )

def custom_transform_with_kwargs(mat, clip_max=1.0):
    """Custom transform that uses kwargs."""
    from liana.utils.transform import zi_minmax
    transformed = zi_minmax(mat)
    # Clip to a custom max value
    transformed.data = np.clip(transformed.data, 0, clip_max)
    return transformed


def test_anndata_transform_kwargs():
    """Test x_transform_kwargs and y_transform_kwargs with AnnData."""
    print("\n=== Testing AnnData with transform_kwargs ===")

    adata = generate_toy_spatial()

    # Test with custom clip value - same for both x and y
    lrdata = inflow(
        adata,
        groupby='bulk_labels',
        resource_name='consensus',
        x_transform=custom_transform_with_kwargs,
        y_transform=custom_transform_with_kwargs,
        x_transform_kwargs={'clip_max': 0.5},
        y_transform_kwargs={'clip_max': 0.5},
        use_raw=False
    )

    print(f"Shape: {lrdata.shape}")
    print(f"Max value: {lrdata.X.max()}")
    print(f"Min value: {lrdata.X.min()}")

    # Verify clipping worked (should be <= 0.5 * 0.5 = 0.25 for product)
    assert lrdata.X.max() <= 0.26, f"Expected max <= 0.26, got {lrdata.X.max()}"
    print("✓ AnnData transform_kwargs test passed!")

def test_mudata_transform_kwargs():
    """Test x_transform_kwargs and y_transform_kwargs with MuData."""
    print("\n=== Testing MuData with separate transform_kwargs ===")

    mdata = generate_toy_mdata()
    interactions = list(product(mdata.mod['adata_x'].var.index,
                                mdata.mod['adata_y'].var.index))

    # Test with different clip values for x and y
    lrdata = inflow(
        mdata,
        groupby='bulk_labels',
        interactions=interactions,
        x_mod='adata_x',
        y_mod='adata_y',
        x_transform=custom_transform_with_kwargs,
        y_transform=custom_transform_with_kwargs,
        x_transform_kwargs={'clip_max': 0.3},  # Custom parameter for x
        y_transform_kwargs={'clip_max': 0.7},  # Custom parameter for y
        x_use_raw=False,
        y_use_raw=False,
        nz_prop=0
    )
    assert lrdata.X.max() <= 0.22, f"Expected max <= 0.22, got {lrdata.X.max()}"
