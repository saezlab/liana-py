from itertools import product

import numpy as np

from liana.method import inflow
from liana.testing._sample_anndata import generate_toy_mdata, generate_toy_spatial


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
