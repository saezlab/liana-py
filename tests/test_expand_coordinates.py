import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from liana.utils import expand_coordinates


def create_test_adata(n_per_sample, n_samples, seed, spatial_key='spatial', sample_key='sample'):
    """
    Helper to create a test AnnData with several samples sharing overlapping coordinates.
    """
    rng = np.random.default_rng(seed)
    coords = np.concatenate([rng.uniform(0, 100, size=(n_per_sample, 2)) for _ in range(n_samples)])
    samples = np.repeat([f'sample_{i}' for i in range(n_samples)], n_per_sample)
    obs = pd.DataFrame({sample_key: pd.Categorical(samples)})
    return AnnData(X=np.zeros((n_per_sample * n_samples, 3)), obs=obs, obsm={spatial_key: coords})


@pytest.fixture
def adata():
    return create_test_adata(n_per_sample=50, n_samples=3, seed=0)


def test_returns_new_object(adata):
    expanded = expand_coordinates(adata, sample_key='sample', n_cols=2)
    assert isinstance(expanded, AnnData)
    assert expanded is not adata


def test_input_is_not_mutated(adata):
    original = adata.obsm['spatial'].copy()
    expand_coordinates(adata, sample_key='sample', n_cols=2)
    np.testing.assert_array_equal(adata.obsm['spatial'], original)
    assert 'spatial_original' not in adata.obsm


def test_original_coordinates_are_preserved(adata):
    expanded = expand_coordinates(adata, sample_key='sample', n_cols=2)
    np.testing.assert_array_equal(expanded.obsm['spatial_original'], adata.obsm['spatial'])


def test_row_order_is_preserved(adata):
    # each row must still map to its own sample after expansion
    expanded = expand_coordinates(adata, sample_key='sample', n_cols=2)
    assert expanded.obs_names.equals(adata.obs_names)
    # within a sample the expansion is a pure translation, so pairwise offsets are unchanged
    for sample in adata.obs['sample'].cat.categories:
        mask = np.asarray(adata.obs['sample'] == sample)
        before = adata.obsm['spatial'][mask]
        after = expanded.obsm['spatial'][mask]
        shift = after - before
        np.testing.assert_allclose(shift - shift[0], 0, atol=1e-9)


def test_grid_layout_prevents_overlap(adata):
    # 3 samples, 2 columns -> [(row=0, col=0), (row=0, col=1), (row=1, col=0)]
    expanded = expand_coordinates(adata, sample_key='sample', n_cols=2)
    coords = expanded.obsm['spatial']
    s = adata.obs['sample'].to_numpy()
    col0, col1, row1 = (coords[s == f'sample_{i}'] for i in range(3))

    assert col1[:, 0].min() > col0[:, 0].max()
    assert row1[:, 1].min() > col0[:, 1].max()


def test_single_sample_shifts_to_origin():
    adata = create_test_adata(n_per_sample=50, n_samples=1, seed=1)
    expanded = expand_coordinates(adata, sample_key='sample')
    np.testing.assert_allclose(expanded.obsm['spatial'].min(axis=0), [0, 0], atol=1e-9)


def test_margin_increases_spacing(adata):
    s = adata.obs['sample'].to_numpy()

    def gap(margin):
        coords = expand_coordinates(adata, sample_key='sample', n_cols=2, margin=margin).obsm['spatial']
        return coords[s == 'sample_1'][:, 0].min() - coords[s == 'sample_0'][:, 0].max()

    assert gap(1.0) > gap(0.0)


def test_custom_spatial_key(adata):
    key = 'custom_coords'
    adata.obsm[key] = adata.obsm.pop('spatial')
    expanded = expand_coordinates(adata, sample_key='sample', spatial_key=key, n_cols=2)
    assert f'{key}_original' in expanded.obsm
    assert 'spatial' not in expanded.obsm


def test_raises_on_missing_sample_key(adata):
    with pytest.raises(ValueError, match='not_a_column'):
        expand_coordinates(adata, sample_key='not_a_column')


def test_raises_on_missing_spatial_key(adata):
    with pytest.raises(ValueError, match='not_a_key'):
        expand_coordinates(adata, sample_key='sample', spatial_key='not_a_key')
