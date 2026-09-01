import numpy as np
import pandas as pd
from anndata import AnnData
from tests._helpers import not_none

from liana.method import cellphonedb, natmi
from liana.preprocessing.spatial_neighbors import spatial_neighbors, spatial_pair_proximity


def test_get_spatial_connectivities(toy_spatial: AnnData) -> None:
    spatial_neighbors(adata=toy_spatial, bandwidth=200, set_diag=True, cutoff=0.2)
    np.testing.assert_equal(
        toy_spatial.obsp["spatial_connectivities"].shape, (toy_spatial.shape[0], toy_spatial.shape[0])
    )
    np.testing.assert_almost_equal(toy_spatial.obsp["spatial_connectivities"].sum(), 4550.654013895928, decimal=4)

    spatial_neighbors(adata=toy_spatial, bandwidth=100, set_diag=True, cutoff=0.1)
    np.testing.assert_almost_equal(toy_spatial.obsp["spatial_connectivities"].sum(), 1802.332962418902, decimal=4)

    conns = not_none(
        spatial_neighbors(adata=toy_spatial, bandwidth=100, kernel="linear", cutoff=0.1, set_diag=True, inplace=False)
    )
    np.testing.assert_almost_equal(conns.sum(), 899.065036633088, decimal=4)

    conns = not_none(
        spatial_neighbors(
            adata=toy_spatial, bandwidth=100, kernel="exponential", cutoff=0.1, set_diag=True, inplace=False
        )
    )
    np.testing.assert_almost_equal(conns.sum(), 1520.8496098963612, decimal=4)

    conns = not_none(
        spatial_neighbors(
            adata=toy_spatial, bandwidth=100, set_diag=True, kernel="misty_rbf", cutoff=0.1, inplace=False
        )
    )
    np.testing.assert_almost_equal(conns.sum(), 1254.3161716188595, decimal=4)

    conns = not_none(
        spatial_neighbors(
            adata=toy_spatial,
            bandwidth=250,
            set_diag=False,
            max_neighbours=100,
            kernel="gaussian",
            cutoff=0.1,
            inplace=False,
        )
    )
    np.testing.assert_almost_equal(conns.sum(), 6597.05237692107, decimal=4)

    conns = not_none(
        spatial_neighbors(
            adata=toy_spatial,
            bandwidth=250,
            set_diag=False,
            max_neighbours=100,
            kernel="gaussian",
            cutoff=0.1,
            inplace=False,
            standardize=True,
        )
    )
    np.testing.assert_almost_equal(conns.sum(), conns.shape[0], decimal=4)


def test_cell_type_proximity_basic() -> None:
    """Test basic proximity calculation between cell types"""
    # Create simple test data with two distinct cell type clusters
    import anndata as ad

    groupby_labels = np.array(["TypeA"] * 10 + ["TypeB"] * 10)
    # TypeA at origin, TypeB far away
    coords_a = np.random.randn(10, 2) * 0.5  # tight cluster at origin
    coords_b = np.random.randn(10, 2) * 0.5 + 100  # tight cluster at (100, 100)
    coordinates = np.vstack([coords_a, coords_b])

    adata = ad.AnnData(np.zeros((20, 10)))
    adata.obs["cell_type"] = groupby_labels
    adata.obsm["spatial"] = coordinates

    proximity_df = spatial_pair_proximity(
        adata=adata, groupby="cell_type", spatial_key="spatial", bandwidth=50, kernel="gaussian"
    )

    # Check output structure
    assert isinstance(proximity_df, pd.DataFrame)
    assert set(proximity_df.columns) == {"source", "target", "mean_distance", "interacting", "proximity"}
    assert len(proximity_df) == 4  # TypeA->TypeA, TypeA->TypeB, TypeB->TypeA, TypeB->TypeB

    # Check self-interactions have high proximity
    self_prox_a = proximity_df[(proximity_df["source"] == "TypeA") & (proximity_df["target"] == "TypeA")][
        "proximity"
    ].to_numpy()[0]
    assert self_prox_a > 0.5  # Should be high for tight cluster

    # Check cross-interactions have low proximity (distant clusters)
    cross_prox = proximity_df[(proximity_df["source"] == "TypeA") & (proximity_df["target"] == "TypeB")][
        "proximity"
    ].to_numpy()[0]
    assert cross_prox < 0.1  # Should be very low for distant clusters


def test_cell_type_proximity_with_contact() -> None:
    """Test optional contact_proximity calculation"""
    import anndata as ad

    groupby_labels = np.array(["TypeA"] * 5 + ["TypeB"] * 5)
    coordinates = np.random.randn(10, 2) * 10

    adata = ad.AnnData(np.zeros((10, 10)))
    adata.obs["cell_type"] = groupby_labels
    adata.obsm["spatial"] = coordinates

    # With contact_bandwidth
    proximity_df = spatial_pair_proximity(
        adata=adata, groupby="cell_type", spatial_key="spatial", bandwidth=100, contact_bandwidth=10, kernel="gaussian"
    )

    assert "contact_proximity" in proximity_df.columns
    assert "contact_interacting" in proximity_df.columns
    assert len(proximity_df) == 4

    # Without contact_bandwidth
    proximity_df_no_contact = spatial_pair_proximity(
        adata=adata,
        groupby="cell_type",
        spatial_key="spatial",
        bandwidth=100,
        contact_bandwidth=None,
        kernel="gaussian",
    )

    assert "contact_proximity" not in proximity_df_no_contact.columns
    assert "contact_interacting" not in proximity_df_no_contact.columns


def test_pipeline_with_spatial_key(pbmc68k: AnnData) -> None:
    """Test that spatial weighting is applied when spatial_key is present"""
    # Add fake spatial coordinates
    np.random.seed(42)
    pbmc68k.obsm["spatial"] = np.random.randn(pbmc68k.shape[0], 2) * 100

    # Run cellphonedb without spatial weighting
    cellphonedb(
        pbmc68k, groupby="bulk_labels", use_raw=True, n_perms=2, key_added="no_spatial", spatial_key="nonexistent"
    )

    # Run cellphonedb with spatial weighting
    cellphonedb(
        pbmc68k, groupby="bulk_labels", use_raw=True, n_perms=2, key_added="with_spatial", spatial_key="spatial"
    )

    # Results should be different (spatial weighting applied)
    res_no_spatial = pbmc68k.uns["no_spatial"]
    res_with_spatial = pbmc68k.uns["with_spatial"]

    assert res_no_spatial.shape == res_with_spatial.shape
    # At least some scores should differ due to proximity weighting
    assert not np.allclose(res_no_spatial["lr_means"].to_numpy(), res_with_spatial["lr_means"].to_numpy())


def test_pipeline_with_spatial_kwargs(pbmc68k: AnnData) -> None:
    """Test that spatial_kwargs are correctly passed through"""
    # Add spatial coordinates with clear clusters
    np.random.seed(123)
    n_cells = pbmc68k.shape[0]
    # Create two well-separated spatial clusters
    cluster1_coords = np.random.randn(n_cells // 2, 2) * 10
    cluster2_coords = np.random.randn(n_cells - n_cells // 2, 2) * 10 + 500
    pbmc68k.obsm["spatial"] = np.vstack([cluster1_coords, cluster2_coords])

    # Run with different bandwidths
    natmi(
        pbmc68k,
        groupby="bulk_labels",
        use_raw=True,
        key_added="short_range",
        spatial_key="spatial",
        spatial_kwargs={"bandwidth": 50, "kernel": "gaussian"},
    )

    natmi(
        pbmc68k,
        groupby="bulk_labels",
        use_raw=True,
        key_added="long_range",
        spatial_key="spatial",
        spatial_kwargs={"bandwidth": 1000, "kernel": "gaussian"},
    )

    # Results should differ based on bandwidth
    res_short = pbmc68k.uns["short_range"]
    res_long = pbmc68k.uns["long_range"]

    assert res_short.shape == res_long.shape
    # Long range should generally have higher scores (less downweighting)
    # Check that at least some scores differ
    assert not np.allclose(res_short["expr_prod"].to_numpy(), res_long["expr_prod"].to_numpy())
