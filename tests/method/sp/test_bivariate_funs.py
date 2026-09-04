from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from liana.method.sp._bivariate._global_functions import _global_r
from liana.method.sp._bivariate._local_functions import (
    LocalFunction,
    LocalStat,
    _local_morans,
    _masked_spearman,
    _norm_product,
    _product,
    _vectorized_cosine,
    _vectorized_jaccard,
    _vectorized_pearson,
    _vectorized_spearman,
)


@pytest.fixture
def mats() -> SimpleNamespace:
    """Two 20x5 feature matrices and a 20x20 weight matrix."""
    rng = np.random.default_rng(seed=0)

    return SimpleNamespace(
        x_mat=rng.normal(size=(20, 5)).astype(np.float32),
        y_mat=rng.normal(size=(20, 5)).astype(np.float32),
        weight=csr_matrix(rng.uniform(size=(20, 20)).astype(np.float32)),
    )


@pytest.fixture
def pval_mats() -> SimpleNamespace:
    """Two 10x10 feature matrices and a weight matrix scaled to sum to `n`."""
    seed = 0
    rng = np.random.default_rng(seed=seed)

    dist = csr_matrix(rng.normal(size=(10, 10)))
    norm_factor = dist.shape[0] / dist.sum()

    return SimpleNamespace(
        seed=seed,
        rng=rng,
        weight=csr_matrix(norm_factor * dist),
        x_mat=rng.normal(size=(10, 10)),
        y_mat=rng.normal(size=(10, 10)),
        n_perms=100,
        mask_negatives=True,
    )


def _assert_bivariate(
    function: LocalStat,
    desired: np.ndarray,
    mats: SimpleNamespace,
    dense_weight: bool = False,
) -> None:
    weight = mats.weight.toarray() if dense_weight else mats.weight
    actual = function(mats.x_mat, mats.y_mat, weight)
    assert actual.shape == (20, 5)
    np.testing.assert_almost_equal(actual[0, :], desired, decimal=5)


# ── local functions ───────────────────────────────────────────────────────────


def test_pc_vectorized(mats: SimpleNamespace) -> None:
    pc_vec_truth = np.array([0.25005114, 0.04262733, -0.00130362, 0.2903336, -0.1236529])
    _assert_bivariate(_vectorized_pearson, pc_vec_truth, mats)


def test_sp_vectorized(mats: SimpleNamespace) -> None:
    sp_vec_truth = np.array([0.23636213, 0.16480759, -0.01487235, 0.22840601, -0.11492937])
    _assert_bivariate(_vectorized_spearman, sp_vec_truth, mats)


def test_sp_masked(mats: SimpleNamespace) -> None:
    sp_masked_truth = np.array([0.23636216, 0.16480756, -0.0148723, 0.22840606, -0.11492944])
    _assert_bivariate(_masked_spearman, sp_masked_truth, mats, dense_weight=True)


def test_costine_vectorized(mats: SimpleNamespace) -> None:
    cosine_vec_truth = np.array([0.33806977, 0.03215113, 0.0950243, 0.2957758, -0.10259595])
    _assert_bivariate(_vectorized_cosine, cosine_vec_truth, mats)


def test_vectorized_jaccard(mats: SimpleNamespace) -> None:
    jaccard_vec_truth = np.array([0.34295967, 0.35367563, 0.39685577, 0.41780996, 0.30527356])
    _assert_bivariate(_vectorized_jaccard, jaccard_vec_truth, mats)


# NOTE: spatialdm uses raw counts
def test_morans(mats: SimpleNamespace) -> None:
    sp_morans_truth = np.array([-1.54256, 0.64591, 1.30025, 0.55437, -0.77182])
    _assert_bivariate(_local_morans, sp_morans_truth, mats)


def test_product(mats: SimpleNamespace) -> None:
    product_vec_truth = np.array([5.4518123, -0.7268728, 8.350364, 0.53861964, 1.4466602])
    _assert_bivariate(_product, product_vec_truth, mats, dense_weight=True)


def test_norm_product(mats: SimpleNamespace) -> None:
    product_vec_truth = np.array([0.4081537, -0.03988646, 0.42921585, 0.03255661, 0.08895018])
    _assert_bivariate(_norm_product, product_vec_truth, mats, dense_weight=True)


# ── p-values ──────────────────────────────────────────────────────────────────


def test_local_permutation_pvals(pval_mats: SimpleNamespace) -> None:
    local_morans = LocalFunction._get_instance("morans")
    local_truth = pval_mats.rng.normal(size=(10, 10))

    pvals = local_morans._permutation_pvals(
        x_mat=pval_mats.x_mat,
        y_mat=pval_mats.y_mat,
        local_truth=local_truth,
        weight=pval_mats.weight,
        n_perms=pval_mats.n_perms,
        seed=pval_mats.seed,
        mask_negatives=pval_mats.mask_negatives,
        verbose=False,
    )
    assert pvals.shape == (10, 10)


def test_local_zscore_pvals(pval_mats: SimpleNamespace) -> None:
    local_morans = LocalFunction._get_instance("morans")
    local_truth = pval_mats.rng.normal(size=(10, 10))

    actual = local_morans._zscore_pvals(
        x_mat=pval_mats.x_mat,
        y_mat=pval_mats.y_mat,
        weight=pval_mats.weight,
        local_truth=local_truth,
        mask_negatives=pval_mats.mask_negatives,
    )
    assert actual.shape == (10, 10)


def test_global_zscore_pvals(pval_mats: SimpleNamespace) -> None:
    global_stat = pval_mats.rng.normal(size=(10))
    pvals = _global_r._zscore_pvals(
        global_stat=global_stat, weight=pval_mats.weight, mask_negatives=pval_mats.mask_negatives
    )
    assert pvals.shape == (10,)


def test_global_permutation_pvals(pval_mats: SimpleNamespace) -> None:
    global_stat = pval_mats.rng.normal(size=(10))
    pvals = _global_r._permutation_pvals(
        x_mat=pval_mats.x_mat,
        y_mat=pval_mats.y_mat,
        global_stat=global_stat,
        seed=pval_mats.seed,
        n_perms=pval_mats.n_perms,
        mask_negatives=pval_mats.mask_negatives,
        weight=pval_mats.weight,
        verbose=False,
    )
    assert pvals.shape == (10,)
