import numpy as np

from liana.plotting import connectivity


def test_proximity_plot(toy_spatial):
    idx = 0
    plot_data = connectivity(adata=toy_spatial, idx=idx).data

    # one point per spot, coloured by its connectivity to `idx`, drawn
    # weakest-first so the strongest neighbours end up on top
    assert plot_data.shape[0] == toy_spatial.shape[0]
    assert plot_data['connectivity'].is_monotonic_increasing

    coords = toy_spatial.obsm['spatial']
    expected = np.asarray(toy_spatial.obsp['spatial_connectivities'][:, idx].todense()).ravel()
    by_spot = plot_data.loc[toy_spatial.obs_names]
    np.testing.assert_allclose(by_spot['connectivity'].values, expected)

    # x is kept as-is, y is flipped so the image reads top-down
    np.testing.assert_array_equal(by_spot['x'].values, coords[:, 0])
    np.testing.assert_array_equal(by_spot['y'].values, coords[:, 1].max() - coords[:, 1])
