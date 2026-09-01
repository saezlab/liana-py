from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm import tqdm

from liana._core._types import get_obs, get_x

if TYPE_CHECKING:
    from anndata import AnnData

    from liana._core._types import MatrixLike


type _AggFn = Callable[..., NDArray[np.floating]]
"""Aggregates a matrix along an ``axis``, e.g. :func:`numpy.mean`.

Spelled with `...` parameters because the callables passed here include numpy
ufunc-style overloads that no single explicit signature matches structurally;
the return type is what callers rely on.
"""


def _get_means_perms(
    adata: AnnData,
    n_perms: int,
    seed: int,
    agg_fn: _AggFn,
    norm_factor: float | np.floating | None,
    n_jobs: int,
    verbose: bool,
) -> NDArray[np.floating]:
    """
    Generate permutations and indices required for permutation-based methods

    Parameters
    ----------
    adata
        Annotated data matrix
    n_perms
        Number of permutations to be calculated
    seed
        Random seed for reproducibility.
    agg_fn
        Function by which to aggregate the matrix, should take `axis` argument
    norm_factor
        Additionally normalize the data by some factor (e.g. matrix max for CellChat)
    n_jobs
        Number of parallel threads to run the analysis.
    verbose
        Verbosity bool

    Returns
    -------
    Tuple with:
        - perms: 3D tensor with permuted averages per cluster
        - ligand_pos: Index of the ligand in the tensor
        - receptor_pos: Index of the receptor in the perms tensor
        - labels_pos: Index of cell identities in the perms tensor

    """
    X = get_x(adata)
    # Gate on the value, not its concrete numpy type: an `isinstance(..., np.float32)`
    # check silently skipped normalisation for a plain `float`.
    if norm_factor is not None:
        # Divide out-of-place (not `/=`) so we don't mutate the caller's matrix -- `adata.X`
        # may share its buffer with `adata.raw.X`. Cast back to the original dtype because
        # sparse `matrix / scalar` promotes to float64, which would otherwise double memory.
        X = (X / norm_factor).astype(X.dtype)
        adata.X = X

    # define labels and masks
    obs = get_obs(adata)
    labels = obs["@label"].cat.categories
    labels_mask = np.zeros((adata.shape[0], labels.shape[0]), dtype=bool)

    # populate masks shape(genes, labels)
    for ct_idx, label in enumerate(labels):
        labels_mask[:, ct_idx] = obs["@label"] == label

    # Perm should be a cube /w dims: n_perms x idents x n_genes
    perms = _generate_perms_cube(X, n_perms, labels_mask, seed, agg_fn, n_jobs, verbose)

    return perms


# Define a helper function for parallel processing
def _permute_and_aggregate(
    perm: int,
    perm_idx: NDArray[np.integer],
    X: MatrixLike,
    labels_mask: NDArray[np.bool_],
    agg_fn: _AggFn,
) -> tuple[int, NDArray[np.floating]]:
    perm_mat = X[perm_idx]
    permuted_means = np.array([agg_fn(perm_mat[labels_mask[:, i]], axis=0) for i in range(labels_mask.shape[1])])
    return perm, permuted_means


def _generate_perms_cube(
    X: MatrixLike,
    n_perms: int,
    labels_mask: NDArray[np.bool_],
    seed: int,
    agg_fn: _AggFn,
    n_jobs: int,
    verbose: bool,
) -> NDArray[np.floating]:
    # initialize rng
    rng = np.random.default_rng(seed=seed)

    # indexes to be shuffled
    idx = np.arange(X.shape[0])

    # Perm should be a cube /w dims: n_perms x idents x n_genes
    perms = np.zeros((n_perms, labels_mask.shape[1], X.shape[1]))

    # Use Parallel to enable parallelization
    results = Parallel(n_jobs=n_jobs)(
        delayed(_permute_and_aggregate)(perm, rng.permutation(idx), X, labels_mask, agg_fn)
        for perm in tqdm(range(n_perms), disable=not verbose)
    )

    # Unpack results
    for perm, permuted_means in results:
        perms[perm] = np.reshape(permuted_means, (labels_mask.shape[1], X.shape[1]))

    return perms


def _get_positions(
    adata: AnnData,
    lr_res: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    labels = get_obs(adata)["@label"].cat.categories

    # get positions of each entity in the matrix
    ligand_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity in lr_res["ligand"]}
    receptor_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity in lr_res["receptor"]}
    labels_pos = {labels[pos]: pos for pos in range(labels.shape[0])}

    return ligand_pos, receptor_pos, labels_pos


def _get_mat_idx(
    adata: AnnData,
    lr_res: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    # convert to indexes
    ligand_pos, receptor_pos, labels_pos = _get_positions(adata, lr_res)

    ligand_idx = lr_res["ligand"].map(ligand_pos)
    receptor_idx = lr_res["receptor"].map(receptor_pos)

    source_idx = lr_res["source"].map(labels_pos)
    target_idx = lr_res["target"].map(labels_pos)

    return ligand_idx, receptor_idx, source_idx, target_idx


def _calculate_pvals(
    lr_truth: NDArray[np.floating],
    perm_stats: NDArray[np.floating] | None,
    _score_fn: _AggFn,
    proximity_weights: NDArray[np.floating] | None = None,
) -> NDArray[np.floating] | None:
    """
    Calculate p-values for a given DataFrame x and permutation statistics

    Parameters
    ----------
    lr_truth
        Observed LR scores, shape (n_interactions,)
    perm_stats
        Permutation statistics, shape (2, n_perms, n_interactions)
    _score_fn
        Function to combine ligand and receptor statistics
    proximity_weights
        Optional spatial proximity weights, shape (n_interactions,)

    Returns
    -------
    P-values for the observed scores

    """
    # calculate p-values
    if perm_stats is not None:
        lr_perm_means = _score_fn(perm_stats, axis=0)

        # Apply proximity weights to both observed and permuted if provided
        # Note: proximity weights, if any, are expected to have been applied
        # to the observed scores (lr_truth) upstream. We also apply them
        # to the permuted statistics here to maintain consistency in the
        # null distribution when spatial structure is part of the signal.
        if proximity_weights is not None:
            # Weight permuted: (n_perms, n_interactions) * (n_interactions,)
            # Broadcasting automatically handles dimension alignment
            lr_perm_means = lr_perm_means * proximity_weights

        n_perms = perm_stats.shape[1]
        pvals: NDArray[np.floating] | None = np.sum(np.greater_equal(lr_perm_means, lr_truth), axis=0) / n_perms
    else:
        pvals = None

    return pvals


def _apply_proximity_weights(
    observed_scores: NDArray[np.floating],
    x: pd.DataFrame,
) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
    """
    Extract proximity weights from DataFrame and apply to observed scores.

    Parameters
    ----------
    observed_scores
        Observed interaction scores, shape (n_interactions,)
    x
        DataFrame with LIANA results, may contain 'proximity' column

    Returns
    -------
    Tuple of (weighted_scores, proximity_weights)
        - weighted_scores: observed scores multiplied by proximity if present, otherwise unchanged
        - proximity_weights: array of weights or None if not present
    """
    proximity_weights = np.asarray(x["proximity"].to_numpy()) if "proximity" in x.columns else None
    if proximity_weights is not None:
        observed_scores = observed_scores * proximity_weights
    return observed_scores, proximity_weights
