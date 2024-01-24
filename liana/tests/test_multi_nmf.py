import numpy as np
from liana.testing._sample_anndata import generate_toy_adata
from liana.multi import nmf, estimate_elbow

adata = generate_toy_adata()
adata.X = np.abs(adata.X)


def test_run_nmf():
    W, H, _, _ = nmf(adata, n_components=2, inplace=False)

    assert W.shape == (adata.n_obs, 2)
    assert H.shape == (adata.n_vars, 2)

    nmf(adata, n_components=None, inplace=True, random_state=0, max_iter=20)
    assert 'NMF_W' in adata.obsm
    assert 'NMF_H' in adata.varm
    assert adata.obsm['NMF_W'].shape == (adata.n_obs, 4)
    assert adata.varm['NMF_H'].shape == (adata.n_vars, 4)
    assert adata.uns['nmf_errors'].shape == (10, 2)
    assert adata.uns['nmf_rank'] == 4

def test_estimate_elbow():
    errors, rank = estimate_elbow(adata.X, k_range=range(1, 10), random_state=0, max_iter=20)
    assert rank == 4
    assert errors.shape == (9, 2)
    assert errors['k'].tolist() == list(range(1, 10))
    np.testing.assert_almost_equal(errors['error'].mean(), 0.3640689)


def test_run_nmf_df():
    df = adata.to_df()
    W, H, errors, n_components = nmf(df=df, n_components=2, inplace=True, random_state=0, max_iter=20)

    assert W.shape == (adata.n_obs, 2)
    assert H.shape == (adata.n_vars, 2)
    assert n_components == 2
    assert errors is None
