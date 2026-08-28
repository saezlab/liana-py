import numpy as np
import pytest
from pandas import read_csv

from liana._core._pipe_utils._get_mean_perms import _get_means_perms, _get_positions
from liana.method.sc._cellphonedb import _mean
from liana.method.sc._liana_pipe import _trimean


@pytest.fixture
def adata(pbmc68k):
    """`pbmc68k_reduced` as the pipe expects it: `.raw` (log-normalised)
    expression in `.X`, labels in `@label`.
    """
    pbmc68k.X = pbmc68k.raw.X
    pbmc68k.obs['@label'] = pbmc68k.obs.bulk_labels

    return pbmc68k


@pytest.fixture
def all_defaults(data_dir):
    """The liana_pipe output obtained with all default parameters."""
    return read_csv(data_dir / "all_defaults.csv", index_col=0)


def test_perms(adata):
    perms = _get_means_perms(adata=adata,
                             norm_factor=None,
                             agg_fn=_mean,
                             n_perms=100,
                             seed=1337,
                             n_jobs=1,
                             verbose=False)

    assert perms.shape == (100, 10, 765)


def test_positions(adata, all_defaults):
    ligand_pos, receptor_pos, labels_pos = _get_positions(adata, all_defaults)

    assert ligand_pos['MIF'] == 740
    assert receptor_pos['CD4'] == 465
    assert labels_pos['Dendritic'] == 9


def test_cellchat_perms(adata):
    mat_max = adata.X.max()

    perms = _get_means_perms(adata=adata,
                             norm_factor=None,
                             agg_fn=_trimean,
                             n_perms=100,
                             seed=1337,
                             n_jobs=1,
                             verbose=False
                             )

    assert perms.shape == (100, 10, 765)

    desired = np.array([33840.83, 36332.442, 34569.577, 33819.275,
                        33809.956, 33785.234, 33844.524, 34986.043,
                        34304.404, 33644.323])
    expected = perms.sum(axis=0).sum(axis=1)

    np.testing.assert_almost_equal(expected, desired, decimal=3)

    perms = _get_means_perms(adata=adata,
                             norm_factor=mat_max,
                             agg_fn=_trimean,
                             n_perms=100,
                             seed=1337,
                             n_jobs=1,
                             verbose=False
                             )
    desired = np.array([5215.107487, 5599.082231, 5327.412358,
                        5211.785598, 5210.349528, 5206.53966,
                        5215.676758, 5391.592763, 5286.547464,
                        5184.824284])
    expected = perms.sum(axis=0).sum(axis=1)

    np.testing.assert_almost_equal(expected, desired, decimal=6)
