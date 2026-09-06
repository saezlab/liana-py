from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Literal

import numba as nb
import numpy as np
import pandas as pd
from fast_array_utils.types import CSBase
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.sparse import csr_array, csr_matrix
from tqdm import tqdm

from liana._core._types import get_obs, get_x

if TYPE_CHECKING:
    from anndata import AnnData

    prange = range

    from liana._core._types import MatrixLike
else:
    prange = nb.prange


type _AggFn = Callable[..., NDArray[np.floating]]
"""Aggregates a matrix along an ``axis``, e.g. :func:`numpy.mean`.

Spelled with `...` parameters because the callables passed here include numpy
ufunc-style overloads that no single explicit signature matches structurally;
the return type is what callers rely on.
"""

type Aggregation = Literal["mean", "trimean"]
"""The location estimate a permutation-based method builds its null from."""

_TIE_RTOL = 1e-6
"""How close a permuted score has to be to the observed one to count as tied with it.

An order of magnitude above the resolution of the `float32` the scores are stored in, and orders below any difference that carries signal.
"""

_MAX_PERM_INDEX_ELEMENTS = 1 << 24
"""Cap on how many row positions one block of permutations may hold, so that peak memory does not scale with ``n_perms``."""


def _trimean(a: CSBase, axis: int = 0) -> NDArray[np.floating]:
    """Tukey's trimean, the location estimate CellChat scores with."""
    dense = a.toarray()
    quantiles = np.quantile(dense, q=[0.25, 0.75], axis=axis)
    median = np.median(dense, axis=axis)
    return np.asarray((quantiles[0] + 2 * median + quantiles[1]) / 4)


_AGG_FNS: dict[Aggregation, _AggFn] = {"mean": np.mean, "trimean": _trimean}


@contextmanager
def _numba_threads(n_jobs: int) -> Iterator[None]:
    """Run the enclosed numba kernels on at most ``n_jobs`` threads, then restore the previous setting."""
    available: int = nb.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    previous = nb.get_num_threads()
    nb.set_num_threads(available if n_jobs <= 0 else min(n_jobs, available))
    try:
        yield
    finally:
        nb.set_num_threads(previous)


def _get_means_perms(
    adata: AnnData,
    n_perms: int,
    seed: int,
    aggregation: Aggregation,
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
    aggregation
        Location estimate to aggregate each permuted cluster by, `'mean'` or `'trimean'`
    norm_factor
        Additionally normalize the data by some factor (e.g. matrix max for CellChat)
    n_jobs
        Number of parallel threads to run the analysis.
    verbose
        Verbosity bool

    Returns
    -------
    A `(n_perms, n_labels, n_genes)` tensor of permuted per-cluster aggregates.
    """
    X = get_x(adata)
    # gating on `isinstance(..., np.float32)` silently skipped a plain `float`
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
    perms = _generate_perms_cube(X, n_perms, labels_mask, seed, aggregation, n_jobs, verbose)

    return perms


@nb.njit(cache=True)
def _lerp(a: float, b: float, t: float) -> float:
    """Interpolate between ``a`` and ``b``, the way :func:`numpy.quantile` does.

    Approaching from whichever end is nearer keeps the result exact at ``t`` of 0 and 1, so the kernel reproduces numpy's quantiles bit for bit.
    """
    diff = b - a
    if t >= 0.5:
        return b - diff * (1.0 - t)
    return a + diff * t


@nb.njit(cache=True)
def _sparse_quantile(values: NDArray[np.floating], n_zeros: int, n_total: int, q: float) -> float:
    """Take the ``q``-th quantile of ``n_zeros`` zeros followed by the ascending ``values``.

    A column of a sparse group is exactly that: its stored entries, plus one zero for every row that stored nothing.
    Sorting only the stored entries makes the cost proportional to the non-zeros rather than to the number of cells.
    """
    pos = q * (n_total - 1)
    lo = int(np.floor(pos))
    hi = lo + 1 if lo + 1 < n_total else n_total - 1
    a = np.float64(0.0) if lo < n_zeros else np.float64(values[lo - n_zeros])
    b = np.float64(0.0) if hi < n_zeros else np.float64(values[hi - n_zeros])
    return _lerp(a, b, pos - lo)


@nb.njit(cache=True)
def _sparse_trimean(values: NDArray[np.floating], n_zeros: int, n_total: int) -> float:
    """Take Tukey's trimean of ``n_zeros`` zeros followed by the ascending ``values``."""
    half = n_total // 2
    if n_total % 2 == 1:
        median = np.float32(0.0) if half < n_zeros else values[half - n_zeros]
    else:
        lower = np.float32(0.0) if half - 1 < n_zeros else values[half - 1 - n_zeros]
        upper = np.float32(0.0) if half < n_zeros else values[half - n_zeros]
        median = np.float32((lower + upper) / np.float32(2.0))

    q25 = _sparse_quantile(values, n_zeros, n_total, 0.25)
    q75 = _sparse_quantile(values, n_zeros, n_total, 0.75)

    return (q25 + 2 * np.float64(median) + q75) / 4


@nb.njit(parallel=True, cache=True)
def _perm_group_sums(
    data: NDArray[np.floating],
    indices: NDArray[np.integer],
    indptr: NDArray[np.integer],
    perm_indices: NDArray[np.integer],
    label_of_row: NDArray[np.integer],
    n_labels: int,
    n_genes: int,
) -> NDArray[np.floating]:
    """Sum the rows of a CSR matrix per label, once per permutation in ``perm_indices``.

    Each permutation reassigns which cell sits at which position, so position ``j`` contributes row ``perm_indices[p, j]`` to the label that position ``j`` originally carried.
    Accumulating straight out of the CSR buffers costs one pass over the non-zeros per permutation and never materialises a permuted copy of the matrix.
    Sums are taken in double precision, where the gathered-rows fallback inherits scipy's single-precision accumulation.
    """
    n_perms, n_obs = perm_indices.shape
    sums = np.zeros((n_perms, n_labels, n_genes), dtype=np.float64)

    for p in prange(n_perms):
        for j in range(n_obs):
            label = label_of_row[j]
            row = perm_indices[p, j]
            for k in range(indptr[row], indptr[row + 1]):
                sums[p, label, indices[k]] += data[k]

    return sums


@nb.njit(parallel=True, cache=True)
def _perm_group_trimeans(
    data: NDArray[np.floating],
    indices: NDArray[np.integer],
    indptr: NDArray[np.integer],
    perm_indices: NDArray[np.integer],
    order: NDArray[np.integer],
    label_ptr: NDArray[np.integer],
    n_genes: int,
) -> NDArray[np.floating]:
    """Take Tukey's trimean of the rows of a CSR matrix per label, once per permutation in ``perm_indices``.

    ``order`` lists row positions grouped by label and ``label_ptr`` delimits those groups.
    A group's non-zeros are bucketed by gene in one counting pass, so each gene's column is sorted on its own instead of densifying the whole group.
    """
    n_perms = perm_indices.shape[0]
    n_labels = label_ptr.shape[0] - 1
    trimeans = np.zeros((n_perms, n_labels, n_genes), dtype=np.float64)

    for p in prange(n_perms):
        offsets = np.empty(n_genes + 1, dtype=np.int64)
        for label in range(n_labels):
            start, stop = label_ptr[label], label_ptr[label + 1]
            n_in_label = stop - start

            offsets[:] = 0
            for position in range(start, stop):
                row = perm_indices[p, order[position]]
                for k in range(indptr[row], indptr[row + 1]):
                    offsets[indices[k] + 1] += 1
            for gene in range(n_genes):
                offsets[gene + 1] += offsets[gene]

            values = np.empty(offsets[n_genes], dtype=data.dtype)
            cursor = offsets[:n_genes].copy()
            for position in range(start, stop):
                row = perm_indices[p, order[position]]
                for k in range(indptr[row], indptr[row + 1]):
                    gene = indices[k]
                    values[cursor[gene]] = data[k]
                    cursor[gene] += 1

            for gene in range(n_genes):
                column = values[offsets[gene] : offsets[gene + 1]]
                column.sort()
                trimeans[p, label, gene] = _sparse_trimean(column, n_in_label - column.size, n_in_label)

    return trimeans


def _permute_and_aggregate(
    perm_indices: NDArray[np.integer],
    X: MatrixLike,
    label_rows: Sequence[NDArray[np.integer]],
    agg_fn: _AggFn,
) -> NDArray[np.floating]:
    """Aggregate ``X`` per label under each permutation in ``perm_indices``.

    The fallback for aggregations that are not a plain mean, and so have no closed-form accumulation.
    Rows are gathered per label rather than by permuting the whole matrix, which selects the same rows in the same order at a fraction of the cost.
    """
    return np.array([[agg_fn(X[perm_idx[rows]], axis=0) for rows in label_rows] for perm_idx in perm_indices])


def _chunk_permutations(
    rng: np.random.Generator,
    n_obs: int,
    n_perms: int,
    n_chunks: int,
) -> Iterator[NDArray[np.integer]]:
    """Draw ``n_perms`` shuffles of ``range(n_obs)``, yielded in blocks.

    Lazily, so that peak memory is set by the block size rather than by ``n_perms``, and in the smallest integer type that fits ``n_obs``.
    """
    dtype = np.min_scalar_type(n_obs - 1)
    idx = np.arange(n_obs)
    for chunk in np.array_split(np.arange(n_perms), n_chunks):
        yield np.array([rng.permutation(idx) for _ in chunk], dtype=dtype)


def _generate_perms_cube(
    X: MatrixLike,
    n_perms: int,
    labels_mask: NDArray[np.bool_],
    seed: int,
    aggregation: Aggregation,
    n_jobs: int,
    verbose: bool,
) -> NDArray[np.floating]:
    """Build the ``(n_perms, n_labels, n_genes)`` cube of per-label aggregates under the null.

    Permutations are drawn in the same order whatever ``n_jobs`` is, so the cube -- and every p-value derived from it -- depends only on ``seed``.
    They are also drawn, aggregated and written out a block at a time, so peak memory is set by the block size rather than by ``n_perms``.
    """
    rng = np.random.default_rng(seed=seed)
    n_labels = labels_mask.shape[1]
    n_genes = X.shape[1]
    label_rows = [np.flatnonzero(labels_mask[:, label]) for label in range(n_labels)]
    counts = np.array([rows.size for rows in label_rows])

    csr = X if isinstance(X, csr_matrix | csr_array) else None
    compiled = csr is not None and _kernel_applies(csr, aggregation)

    n_chunks = max(1, min(n_perms, -(-n_perms * X.shape[0] // _MAX_PERM_INDEX_ELEMENTS)))
    if not compiled and n_jobs != 1:
        n_chunks = max(n_chunks, min(n_perms, 4 * n_jobs))
    chunks = _chunk_permutations(rng, X.shape[0], n_perms, n_chunks)

    results: Iterable[NDArray[np.floating]]
    if csr is not None and compiled:
        data, indices, indptr = _csr_buffers(csr)
        if aggregation == "mean":
            label_of_row = np.argmax(labels_mask, axis=1).astype(np.int64)
            divisor = counts[None, :, None].astype(np.float64)
            results = (
                _perm_group_sums(data, indices, indptr, perm_indices, label_of_row, n_labels, n_genes) / divisor
                for perm_indices in chunks
            )
        else:
            order = np.concatenate(label_rows).astype(np.int64)
            label_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            results = (
                _perm_group_trimeans(data, indices, indptr, perm_indices, order, label_ptr, n_genes)
                for perm_indices in chunks
            )
    elif n_jobs == 1:
        results = (
            _permute_and_aggregate(perm_indices, X, label_rows, _AGG_FNS[aggregation]) for perm_indices in chunks
        )
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            delayed(_permute_and_aggregate)(perm_indices, X, label_rows, _AGG_FNS[aggregation])
            for perm_indices in chunks
        )

    perms = np.zeros((n_perms, n_labels, n_genes))
    offset = 0
    with _numba_threads(n_jobs) if compiled else nullcontext(), tqdm(total=n_perms, disable=not verbose) as bar:
        for chunk_means in results:
            n_chunk = chunk_means.shape[0]
            perms[offset : offset + n_chunk] = np.reshape(chunk_means, (n_chunk, n_labels, n_genes))
            offset += n_chunk
            bar.update(n_chunk)

    return perms


def _csr_buffers(X: csr_matrix | csr_array) -> tuple[NDArray[np.floating], NDArray[np.integer], NDArray[np.integer]]:
    """Hand the CSR triple to a kernel as plain arrays, which scipy types more loosely than it stores them."""
    return np.asarray(X.data), np.asarray(X.indices), np.asarray(X.indptr)


def _kernel_applies(X: csr_matrix | csr_array, aggregation: Aggregation) -> bool:
    """Say whether the compiled kernel for ``aggregation`` may be used on ``X``.

    The trimean kernel treats every row that stored nothing for a gene as a zero that sorts below the stored entries, which a negative entry would violate.
    """
    return not (aggregation == "trimean" and X.data.size and np.asarray(X.data).min() < 0)


def _index_of(index: pd.Index, values: pd.Series, what: str) -> NDArray[np.intp]:
    """Locate ``values`` in ``index``, raising if any of them is absent.

    :meth:`~pandas.Index.get_indexer` hashes the index once; the equivalent :func:`numpy.where` scan per value is quadratic in the number of interactions.
    """
    positions = index.get_indexer(pd.Index(values))
    if (missing := positions == -1).any():
        absent = pd.unique(pd.Series(values)[missing])
        raise KeyError(f"{what}(s) absent from `adata.var_names`: {', '.join(map(str, absent))}.")
    return np.asarray(positions, dtype=np.intp)


def _get_mat_idx(
    adata: AnnData,
    lr_res: pd.DataFrame,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    labels = get_obs(adata)["@label"].cat.categories

    ligand_idx = _index_of(adata.var_names, lr_res["ligand"], "ligand")
    receptor_idx = _index_of(adata.var_names, lr_res["receptor"], "receptor")

    source_idx = _index_of(labels, lr_res["source"], "source label")
    target_idx = _index_of(labels, lr_res["target"], "target label")

    return ligand_idx, receptor_idx, source_idx, target_idx


def _calculate_pvals(
    lr_truth: NDArray[np.floating],
    perm_stats: NDArray[np.floating] | None,
    _score_fn: _AggFn,
) -> NDArray[np.floating] | None:
    """
    Calculate p-values for a given DataFrame x and permutation statistics

    Parameters
    ----------
    lr_truth
        Observed LR scores, shape (n_interactions,). Already proximity-weighted when the
        caller asked for spatial weighting.
    perm_stats
        Permutation statistics, shape (2, n_perms, n_interactions)
    _score_fn
        Function to combine ligand and receptor statistics

    Returns
    -------
    P-values for the observed scores
    """
    if perm_stats is None:
        return None

    lr_perm_means = _score_fn(perm_stats, axis=0)
    n_perms = perm_stats.shape[1]

    # A permutation that leaves a cluster's membership unchanged -- the only possibility
    # when a sample carries a single cluster -- has to score exactly as the observation
    # does. The two sides are accumulated by different routines, so that tie survives only
    # if it is compared within the tolerance of the single precision they are stored in.
    exceeds = np.greater_equal(lr_perm_means, lr_truth) | np.isclose(lr_perm_means, lr_truth, rtol=_TIE_RTOL, atol=0.0)

    return np.asarray(np.sum(exceeds, axis=0) / n_perms)


def _apply_proximity_weights(
    observed_scores: NDArray[np.floating],
    x: pd.DataFrame,
) -> NDArray[np.floating]:
    """Scale observed interaction scores by spatial proximity, where it was computed.

    Only the observed statistic is weighted.
    Weighting the permuted null by the same factor would cancel out of the comparison ``perm * w >= obs * w`` and leave every p-value unchanged, which is what made spatial weighting a no-op for the permutation-based methods.
    Down-weighting only the observed side is what makes a distant pair need a correspondingly stronger expression signal to clear the null.
    """
    if "proximity" not in x.columns:
        return observed_scores
    return observed_scores * np.asarray(x["proximity"].to_numpy())
