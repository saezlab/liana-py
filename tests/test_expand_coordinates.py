import numpy as np
import pytest
from anndata import AnnData

from liana.utils._expand_coordinates import _expand_coordinates


def create_test_adata(n_obs, seed, spatial_key='spatial'):
    """
    Helper function to create a test AnnData object with random spatial coordinates.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n_obs, 2))
    return AnnData(X=np.zeros((n_obs, 3)), obsm={spatial_key: coords})


@pytest.fixture
def adatas():
    return [create_test_adata(n_obs=50, seed=seed) for seed in range(3)]


def test_returns_new_objects(adatas):
    expanded = _expand_coordinates(adatas, n_cols=2)
    assert len(expanded) == len(adatas)
    assert all(isinstance(adata, AnnData) for adata in expanded)
    assert all(a is not b for a, b in zip(adatas, expanded))


def test_inputs_are_not_mutated(adatas):
    originals = [adata.obsm['spatial'].copy() for adata in adatas]
    _expand_coordinates(adatas, n_cols=2)
    for adata, original in zip(adatas, originals):
        np.testing.assert_array_equal(adata.obsm['spatial'], original)


def test_original_coordinates_are_preserved(adatas):
    expanded = _expand_coordinates(adatas, n_cols=2)
    for original, adjusted in zip(adatas, expanded):
        np.testing.assert_array_equal(adjusted.obsm['spatial_original'], original.obsm['spatial'])


def test_grid_layout_prevents_overlap(adatas):
    # 3 samples, 2 columns -> layout is [(row=0, col=0), (row=0, col=1), (row=1, col=0)]
    expanded = _expand_coordinates(adatas, n_cols=2)
    col0, col1, row1 = (adata.obsm['spatial'] for adata in expanded)

    assert col1[:, 0].min() > col0[:, 0].max()
    assert row1[:, 1].min() > col0[:, 1].max()


def test_single_column_stacks_vertically(adatas):
    expanded = _expand_coordinates(adatas, n_cols=1)
    for upper, lower in zip(expanded, expanded[1:]):
        assert lower.obsm['spatial'][:, 1].min() > upper.obsm['spatial'][:, 1].max()


def test_margin_increases_spacing(adatas):
    small_margin = _expand_coordinates(adatas, n_cols=2, margin=0.0)
    large_margin = _expand_coordinates(adatas, n_cols=2, margin=1.0)

    gap_small = small_margin[1].obsm['spatial'][:, 0].min() - small_margin[0].obsm['spatial'][:, 0].max()
    gap_large = large_margin[1].obsm['spatial'][:, 0].min() - large_margin[0].obsm['spatial'][:, 0].max()

    assert gap_large > gap_small


def test_custom_spatial_key(adatas):
    key = 'custom_coords'
    for adata in adatas:
        adata.obsm[key] = adata.obsm.pop('spatial')

    expanded = _expand_coordinates(adatas, spatial_key=key, n_cols=2)
    assert all(f'{key}_original' in adata.obsm for adata in expanded)
    assert all('spatial' not in adata.obsm for adata in expanded)
