import numpy as np
from scipy.sparse import csr_matrix

from liana.method.sp._bivariate._global_functions import _global_r
from liana.method.sp._bivariate._local_functions import LocalFunction
from liana.testing._sample_anndata import generate_toy_spatial
from liana.utils.spatial_neighbors import spatial_neighbors

adata = generate_toy_spatial()

def test_get_spatial_connectivities():
    spatial_neighbors(adata=adata, bandwidth=200, set_diag=True, cutoff=0.2)
    np.testing.assert_equal(adata.obsp['spatial_connectivities'].shape, (adata.shape[0], adata.shape[0]))
    np.testing.assert_almost_equal(adata.obsp['spatial_connectivities'].sum(), 4550.654013895928, decimal=4)

    spatial_neighbors(adata=adata, bandwidth=100, set_diag=True, cutoff=0.1)
    np.testing.assert_almost_equal(adata.obsp['spatial_connectivities'].sum(), 1802.332962418902, decimal=4)

    conns = spatial_neighbors(adata=adata, bandwidth=100,
                              kernel='linear', cutoff=0.1,
                              set_diag=True, inplace=False)
    np.testing.assert_almost_equal(conns.sum(), 899.065036633088, decimal=4)

    conns = spatial_neighbors(adata=adata, bandwidth=100,
                              kernel='exponential', cutoff=0.1,
                              set_diag=True, inplace=False)
    np.testing.assert_almost_equal(conns.sum(), 1520.8496098963612, decimal=4)

    conns = spatial_neighbors(adata=adata, bandwidth=100, set_diag=True,
                              kernel='misty_rbf', cutoff=0.1,
                              inplace=False)
    np.testing.assert_almost_equal(conns.sum(), 1254.3161716188595, decimal=4)

    conns = spatial_neighbors(adata=adata, bandwidth=250, set_diag=False,
                              max_neighbours=100,
                              kernel='gaussian', cutoff=0.1,
                              inplace=False)
    np.testing.assert_almost_equal(conns.sum(), 6597.05237692107, decimal=4)

    conns = spatial_neighbors(adata=adata, bandwidth=250,
                              set_diag=False, max_neighbours=100,
                              kernel='gaussian', cutoff=0.1,
                              inplace=False, standardize=True)
    np.testing.assert_almost_equal(conns.sum(), conns.shape[0], decimal=4)

# toy test data
seed = 0
rng = np.random.default_rng(seed=seed)
dist = csr_matrix(rng.normal(size=(10, 10)))

norm_factor = dist.shape[0] / dist.sum()
weight = csr_matrix(norm_factor * dist)

x_mat = rng.normal(size=(10, 10))
y_mat = rng.normal(size=(10, 10))
n_perms = 100
mask_negatives = True


local_morans = LocalFunction._get_instance('morans')
def test_local_permutation_pvals():
    local_truth = rng.normal(size=(10, 10))
    mask_negatives = True

    pvals = local_morans._permutation_pvals(x_mat = x_mat,
                                            y_mat = y_mat,
                                            local_truth = local_truth,
                                            weight = weight,
                                            n_perms = n_perms,
                                            seed = seed,
                                            mask_negatives=mask_negatives,
                                            verbose=False
                                            )
    assert pvals.shape == (10, 10)


def test_local_zscore_pvals():
    local_truth = rng.normal(size=(10, 10))
    actual = local_morans._zscore_pvals(x_mat=x_mat,
                                        y_mat=y_mat,
                                        weight=weight,
                                        local_truth=local_truth,
                                        mask_negatives=mask_negatives)
    assert actual.shape == (10, 10)



def test_global_zscore_pvals():
    global_stat = rng.normal(size=(10))
    pvals = _global_r._zscore_pvals(global_stat=global_stat,
                                    weight=weight,
                                    mask_negatives=mask_negatives
    )
    assert pvals.shape == (10,)


def test_global_permutation_pvals():
    global_stat = rng.normal(size=(10))
    pvals = _global_r._permutation_pvals(x_mat=x_mat,
                                         y_mat=y_mat,
                                         global_stat=global_stat,
                                         seed=seed,
                                         n_perms=n_perms,
                                         mask_negatives=mask_negatives,
                                         weight=weight,
                                         verbose=False
                                      )
    assert pvals.shape == (10, )
