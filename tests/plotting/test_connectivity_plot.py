import numpy as np
from anndata import AnnData
from tests._helpers import get_obsp
from tests._helpers import plot_data as _frame

from liana.plotting import connectivity


def test_proximity_plot(toy_spatial: AnnData) -> None:
    idx = 0
    plot_data = _frame(connectivity(adata=toy_spatial, idx=idx))

    # one point per spot, coloured by its connectivity to `idx`, drawn
    # weakest-first so the strongest neighbours end up on top
    assert plot_data.shape[0] == toy_spatial.shape[0]
    assert plot_data["connectivity"].is_monotonic_increasing

    coords = toy_spatial.obsm["spatial"]
    connectivities = get_obsp(toy_spatial, "spatial_connectivities")
    expected = np.asarray(connectivities[:, idx]).ravel()
    by_spot = plot_data.loc[toy_spatial.obs_names]
    np.testing.assert_allclose(by_spot["connectivity"].to_numpy(), expected)

    # x is kept as-is, y is flipped so the image reads top-down
    np.testing.assert_array_equal(by_spot["x"].to_numpy(), coords[:, 0])
    np.testing.assert_array_equal(by_spot["y"].to_numpy(), coords[:, 1].max() - coords[:, 1])
