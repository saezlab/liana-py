from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree
from tqdm import tqdm

from liana._constants import DefaultValues as V
from liana._constants import Keys as K
from liana._constants import PrimaryColumns as P
from liana._docs import d
from liana._logging import _logg
from liana.method._pipe_utils import assert_covered, prep_check_adata
from liana.method._pipe_utils._common import _get_groupby_subset
from liana.method.sp._utils import _add_complexes_to_var
from liana.resource.select_resource import _handle_resource

_EDGE_BLOCK_ELEMS = 1 << 22
_MIN_CELLS_FRAC = 0.01

# ── helpers ───────────────────────────────────────────────────────────

def _default_min_cells(adata: AnnData, min_cells: int | None, verbose: bool) -> int:
    """Default ``min_cells`` to an abundance-relative threshold.

    ``None`` means "drop cell types making up no more than ``_MIN_CELLS_FRAC``
    of the slide". ``prep_check_adata`` keeps types with ``count >= min_cells``,
    so the threshold is ``floor(frac * N) + 1``: a type sitting exactly at the
    fraction is dropped. An explicit integer passes through untouched.
    """
    if min_cells is not None:
        return min_cells
    default = int(np.floor(_MIN_CELLS_FRAC * adata.n_obs)) + 1
    _logg(
        f"`min_cells=None`: dropping cell types with <= {_MIN_CELLS_FRAC:.1%} of "
        f"{adata.n_obs} cells (min_cells={default}).",
        verbose=verbose,
    )
    return default

def _linear_transform(expr: np.ndarray) -> np.ndarray:
    """Mean-normalise to a mean of 1"""
    mean = expr.mean(axis=0, keepdims=True)
    smean = np.where(mean > 0, mean, 1.0)
    return expr / smean

def _to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)

def _get_expr(
    adata: AnnData,
    gene_names: list[str],
) -> np.ndarray:
    # `use_raw`/`layer` are already resolved into `.X` by `prep_check_adata`
    return _to_dense(adata[:, gene_names].X)

def _pair_weights(
    adata: AnnData,
    gene_names: list[str],
    idx: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Per-pair expression weights.

    Transform the unique-gene matrix, then gather to per-pair columns ``idx``;
    returns ``(n_cells, n_pairs)``.
    """
    return transform(_get_expr(adata, gene_names))[:, idx]

def _index_resource(
    adata: AnnData,
    resource: pd.DataFrame,
    lr_sep: str,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """
    Filter resource to genes present in ``adata.var_names`` and build integer indices.

    Returns
    -------
    lr_pairs : (n_pairs, 2) int array indexing into ``unique_ligands`` / ``unique_receptors``
    unique_ligands : list of str
    unique_receptors : list of str
    pair_names : list of ``"ligand<lr_sep>receptor"`` labels
    """
    var_set = set(map(str, adata.var_names))
    unique_ligands = [g for g in dict.fromkeys(resource["ligand"]) if g in var_set]
    unique_receptors = [g for g in dict.fromkeys(resource["receptor"]) if g in var_set]
    resource_f = resource[
        resource["ligand"].isin(unique_ligands) & resource["receptor"].isin(unique_receptors)
    ].copy()
    lig_to_idx = {g: i for i, g in enumerate(unique_ligands)}
    rec_to_idx = {g: i for i, g in enumerate(unique_receptors)}
    lr_pairs = np.column_stack([
        resource_f["ligand"].map(lig_to_idx).values,
        resource_f["receptor"].map(rec_to_idx).values,
    ]).astype(int)
    pair_names = (
        resource_f["ligand"].astype(str) + lr_sep + resource_f["receptor"].astype(str)
    ).tolist()
    return lr_pairs, unique_ligands, unique_receptors, pair_names

def _make_radii(
    max_radius: float, radius_step: float, annulus_width: float, extend_first_annulus: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    radii_inner = np.arange(radius_step, max_radius + radius_step, radius_step, dtype=float)
    radii_outer = radii_inner + annulus_width
    if extend_first_annulus:
        radii_inner[0] = 0.0  # merge the [0, radius_step) contact band into the first annulus
    return radii_inner, radii_outer

def _expr_prop_mask(
    mat: np.ndarray, prop_snd: np.ndarray, prop_rcv: np.ndarray, expr_prop: float
) -> np.ndarray:
    """NaN out pairs whose sender or receiver expressing-cell proportion is below the threshold."""
    if expr_prop <= 0:
        return mat
    filt = (prop_snd < expr_prop) | (prop_rcv < expr_prop)
    if filt.any():
        mat = mat.copy()
        mat[:, filt] = np.nan
    return mat

def _bin_pair_counts(
    tree_a: cKDTree, tree_b: cKDTree, radii_inner: np.ndarray, radii_outer: np.ndarray
) -> np.ndarray:
    """Ordered pair count (a in ``tree_a``, b in ``tree_b``) per annulus bin.
    Self-pairs (distance 0) cancel exactly in the ``outer - inner`` difference.
    """
    cum_o = np.asarray(tree_a.count_neighbors(tree_b, radii_outer), dtype=np.float64)
    cum_i = np.asarray(tree_a.count_neighbors(tree_b, radii_inner), dtype=np.float64)
    return cum_o - cum_i #to get count within an annulus

#for pairwise LRIC
def _support_edge_list(
    tree: cKDTree, radii_inner: np.ndarray, radii_outer: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global (i, j, bin) edge list for all ordered pairs within ``radii_outer[-1]``,
    self-pairs excluded, binned via the same half-open ``[inner, outer)`` convention."""
    n_bins = len(radii_inner)
    spdm = tree.sparse_distance_matrix(tree, max_distance=float(radii_outer[-1]), output_type="coo_matrix")
    I, J, D = spdm.row, spdm.col, spdm.data.astype(np.float32)
    m = I != J #remove self-pairs
    I, J, D = I[m], J[m], D[m]
    bin_idx = np.searchsorted(radii_outer, D, side="right") #map distances to bins
    clipped = np.minimum(bin_idx, n_bins - 1)
    valid = (bin_idx < n_bins) & (D >= radii_inner[clipped])
    return I[valid], J[valid], bin_idx[valid]

# LR pair weighting
def _weighted_support_numerator(
    I: np.ndarray,
    J: np.ndarray,
    bin_idx: np.ndarray,
    WL: np.ndarray,
    WR: np.ndarray,
    n_bins: int,
    N: int,
    pair_chunk: int,
) -> np.ndarray:
    """Weighted pair-sum per radius bin: ``sum_{(i,j) in bin b} WL[i,:] * WR[j,:]``.
    Returns ``(n_bins, n_pairs)`` float64.
    """
    n_pairs = WL.shape[1]
    num = np.zeros((n_bins, n_pairs), dtype=np.float64)
    for b in range(n_bins):
        mb = bin_idx == b
        if not mb.any():
            continue
        Ab = sparse.csr_matrix((np.ones(mb.sum(), np.float32), (I[mb], J[mb])), shape=(N, N))
        for p0 in range(0, n_pairs, pair_chunk):
            p1 = min(p0 + pair_chunk, n_pairs)
            num[b, p0:p1] = (WL[:, p0:p1] * (Ab @ WR[:, p0:p1])).sum(axis=0) 
    return num

def _edge_group_bounds(group_key_sorted: np.ndarray, n_groups: int) -> np.ndarray:
    """Start offsets of every group in a group-key-sorted edge list.
    ``bounds[g]:bounds[g + 1]`` is the (possibly empty) contiguous slice of edges
    belonging to group ``g``; ``bounds`` has length ``n_groups + 1``.
    """
    return np.searchsorted(group_key_sorted, np.arange(n_groups + 1), side="left")

#weighted ligand–receptor sums for grouped edge segments
# g -> group, one radius bin for a fixed sender→receiver cell-type pair.
# p -> ligand–receptor pair.
# i→j is a directed spatial edge.
def _segment_weighted_sums(
    I_sorted: np.ndarray,
    J_sorted: np.ndarray,
    bounds: np.ndarray,
    WL: np.ndarray,
    WR: np.ndarray,
    pair_chunk: int,
) -> np.ndarray:
    """``out[g, p] = sum_{(i, j) in group g} WL[i, p] * WR[j, p]``, shape ``(n_groups, n_pairs)``.

    Edges must already be sorted by group key (with ``bounds`` from
    :func:`_edge_group_bounds`) so that each group occupies a contiguous slice.
    Cost is ``O(n_edges_in_these_groups * n_pairs)`` with no dependence on
    ``N`` -- only the cells a group's edges actually reference are touched, so
    a group holding few edges costs proportionally little. Accumulation is
    float64.
    """
    n_groups = len(bounds) - 1
    n_pairs = WL.shape[1]
    out = np.zeros((n_groups, n_pairs), dtype=np.float64)
    for p0 in range(0, n_pairs, pair_chunk):
        p1 = min(p0 + pair_chunk, n_pairs)
        WLc, WRc = WL[:, p0:p1], WR[:, p0:p1]
        # cap the gathered block so peak memory is bounded by _EDGE_BLOCK_ELEMS
        edge_block = max(1, _EDGE_BLOCK_ELEMS // (p1 - p0))
        for g in range(n_groups):
            lo, hi = int(bounds[g]), int(bounds[g + 1])
            if hi <= lo:
                continue
            acc = out[g, p0:p1]
            for e0 in range(lo, hi, edge_block):
                e1 = min(e0 + edge_block, hi)
                acc += (WLc[I_sorted[e0:e1]] * WRc[J_sorted[e0:e1]]).sum(axis=0, dtype=np.float64)
    return out

def _type_mean_weights(
    W: np.ndarray, obs_types: np.ndarray, cell_types_list: list[str]
) -> np.ndarray:
    """Per-type mean weight, shape ``(n_types, n_pairs)``.

    These are the ``w̄L^S`` / ``w̄R^R`` of the conditional null -- the quantity
    that makes a type-level marginal shift cancel exactly. Computed once
    (``O(N * n_pairs)``) and reused by every directed pair involving the type.
    """
    out = np.empty((len(cell_types_list), W.shape[1]), dtype=np.float64)
    for ci, ct in enumerate(cell_types_list):
        out[ci] = W[obs_types == ct].mean(axis=0)
    return out

def _null_pair_product(si: int, ri: int, mL: np.ndarray, mR: np.ndarray) -> np.ndarray:
    """``E[wL_i wR_j]`` for a random ordered S->R pair (S != R) under the
    within-type weight permutation: exactly ``w̄L^S * w̄R^R``, since the
    within-S and within-R permutations are independent."""
    return mL[si] * mR[ri]


# ── CrossPCF ───────────────────────────────────────────

class CrossPCF:

    """Cross pair-correlation function (cross-PCF) between cell types.

    The cross-PCF, ``g(r)``, measures whether sender-type and receiver-type cells
    co-localize at distance ``r`` more or less often than expected under
    **random labelling**: the observed cell locations are held fixed while cell-type
    labels are assumed to be randomly assigned to those locations. Equivalently,
    ``g(r)`` compares the observed number of sender→receiver pairs in each distance
    annulus with the number expected if cell identities were independent of
    position on the observed tissue support.

    ``g(r) > 1`` indicates that the two cell types are spatially co-localised at
    distance ``r`` beyond what would be expected from the tissue architecture
    alone, ``g(r) < 1`` indicates spatial avoidance, and ``g(r) ≈ 1`` indicates
    that cell-type labels are spatially independent given the observed point
    pattern.

    Unlike the classical cross-PCF (Bull et al., 2024, doi:10.1101/2024.12.06.627195),
    which normalises against complete spatial randomness (CSR), this implementation uses 
    a closed-form random-labelling null.
    By conditioning on the observed tissue support, it naturally accounts for
    non-convex tissue boundaries, holes, and fragmentation without estimating an
    effective tissue area.
    """

    @d.dedent
    def __call__(
        self,
        adata: AnnData,
        groupby: str,
        spatial_key: str = K.spatial_key,
        cell_types: Iterable[str] | None = None,
        min_cells: int | None = None,
        max_radius: float = 200,
        radius_step: float = 20,
        annulus_width: float = 20,
        extend_first_annulus: bool = True,
        key_added: str = "cross_pcf",
        inplace: bool = V.inplace,
        verbose: bool = V.verbose,
    ) -> dict | None:
        """Cross pair-correlation function (cross-PCF) between cell types.

        Computes the distance-resolved cross-PCF ``g(r)`` for every directed
        sender->receiver combination of cell types present in
        ``adata.obs[groupby]``, normalised against the empirical any-cell-type
        pair count within this tissue (a random-labelling null). Expression is 
        not used, this characterises tissue architecture alone.

        Parameters
        ----------
        %(adata)s
        %(groupby)s
        %(spatial_key)s
        %(cell_types)s
        %(min_cells)s
            Default ``None`` derives the threshold from slide composition
            instead of using a fixed count: cell types making up no more than
            1% of all cells are dropped.
        %(max_radius)s
        %(radius_step)s
        %(annulus_width)s
        %(extend_first_annulus)s
        %(key_added)s
        %(inplace)s
        %(verbose)s

        Returns
        -------
        dict with keys ``cell_types``, ``radii``, ``results`` if
        ``inplace=False``, else ``None`` (stored in ``adata.uns[key_added]``).
        ``results`` maps ``(sender, receiver)`` tuples to ``(n_bins,)`` arrays.
        """
        _adata_orig = adata
        adata = prep_check_adata(
            adata=adata,
            groupby=groupby,
            min_cells=_default_min_cells(adata, min_cells, verbose),
            groupby_subset=cell_types,
            use_raw=False,
            layer=None,
            obsm={spatial_key: adata.obsm[spatial_key]},
            complex_sep=None,
            verbose=verbose,
        )

        obs_types = adata.obs[groupby].astype(str).values
        cell_types_list = sorted(np.unique(obs_types).tolist())
        pairs = [(s, r) for s in cell_types_list for r in cell_types_list if s != r]
        _logg(
            f"Computing cross-PCF for {len(cell_types_list)} cell types "
            f"({len(pairs)} directed pairs).",
            verbose=verbose,
        )

        coords = np.asarray(adata.obsm[spatial_key], dtype=float)
        N = len(coords)
        radii_inner, radii_outer = _make_radii(
            max_radius, radius_step, annulus_width, extend_first_annulus
        )

        tree_all = cKDTree(coords)
        T = _bin_pair_counts(tree_all, tree_all, radii_inner, radii_outer)  # (n_bins,) whole-tissue support

        counts = {ct: int((obs_types == ct).sum()) for ct in cell_types_list}
        ct_trees = {ct: cKDTree(coords[obs_types == ct]) for ct in cell_types_list}

        results: dict[tuple[str, str], np.ndarray] = {}
        for sender, receiver in tqdm(pairs, disable=not verbose, desc="cross-PCF"):
            O = _bin_pair_counts(ct_trees[sender], ct_trees[receiver], radii_inner, radii_outer)
            with np.errstate(divide="ignore", invalid="ignore"):
                g = O * (N * (N - 1)) / (counts[sender] * counts[receiver] * T)
                g[T == 0] = np.nan
            results[(sender, receiver)] = g

        res = {"cell_types": cell_types_list, "radii": radii_inner, "results": results}
        if inplace:
            _adata_orig.uns[key_added] = res
            return None
        return res

cross_pcf = CrossPCF()

# ── LRIC ───────────────────────────────────────────────

class LRIC:
    """
    Ligand-Receptor Interaction Correlation (LRIC).

    LRIC is an expression-weighted cross pair-correlation function: each cell's
    contribution at distance ``r`` is weighted by its ligand (sender) and
    receptor (receiver) expression. The resulting ``g(r)`` therefore measures
    whether ligand- and receptor-expressing cells are spatially co-enriched at
    distance ``r`` beyond what would be expected under a random-labelling null,
    in which cell locations are held fixed while ligand and receptor weights are
    randomly reassigned among those locations. In pairwise mode that reassignment
    is restricted to within each cell type, so a type's own mean expression level
    is preserved by construction and cannot masquerade as spatial signal.

    ``g(r) > 1`` indicates that ligand- and receptor-expressing cells occur
    together more often than expected from the observed tissue architecture and
    cell-type composition alone, ``g(r) ≈ 1`` indicates that expression provides
    no additional spatial enrichment beyond the underlying tissue structure, and
    ``g(r) < 1`` indicates depletion. This surfaces candidate ligand-receptor
    interactions that are both spatially proximal and co-expressed—a more
    specific signal of potential cell-cell communication than spatial
    co-localisation or co-expression considered alone.

    LRIC builds on the cross pair-correlation function; see
    (:class:`CrossPCF`).

    """

    @d.dedent
    def __call__(
        self,
        adata: AnnData,
        resource: pd.DataFrame | None = V.resource,
        resource_name: str | None = None,
        interactions: list | None = V.interactions,
        groupby: str | None = None,
        spatial_key: str = K.spatial_key,
        max_radius: float = 200,
        radius_step: float = 20,
        annulus_width: float = 20,
        extend_first_annulus: bool = True,
        cell_types: Iterable[str] | None = None,
        min_cells: int | None = None,
        groupby_pairs: pd.DataFrame | None = V.groupby_pairs,
        expr_prop: float = 0.0,
        complex_sep: str | None = V.complex_sep,
        lr_sep: str = V.lr_sep,
        transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        use_raw: bool = V.use_raw,
        layer: str | None = V.layer,
        pair_chunk: int = 256,
        key_added: str = "lric",
        inplace: bool = V.inplace,
        verbose: bool = V.verbose,
    ) -> dict | None:
        """

        Ligand-Receptor Interaction Correlation (LRIC).

        Computes an expression-weighted cross-PCF ``g(r)``: each cell's
        contribution at distance ``r`` is weighted by its ligand and receptor
        expression, so ``g(r) > 1`` flags ligand- and receptor-expressing
        cells that are spatially co-enriched beyond cell-type co-localisation
        alone — candidate ligand-receptor interactions that are both proximal
        and co-expressed.

        When ``groupby`` is ``None`` (default), all cells are treated as
        potential senders and receivers (self-pairs excluded), providing a
        global screen for LR pairs with strong spatial co-enrichment signal.
        Result dict keys: ``pair_names``, ``radii``, ``lric``
        (shape ``(n_bins, n_pairs)``).

        When ``groupby`` is a column name in ``adata.obs``, the LRIC is
        computed for every directed sender→receiver cell-type pair.
        Result dict keys: ``cell_types``, ``pair_names``, ``radii``,
        ``results`` mapping ``(sender, receiver)`` tuples to
        ``(n_bins, n_pairs)`` arrays.

        Parameters
        ----------
        %(adata)s
        %(resource)s
        %(resource_name)s
        %(interactions)s
        groupby
            Column in ``adata.obs`` used to define cell types. ``None`` runs
            the cell-type-agnostic mode across all cells.
        %(spatial_key)s
        %(max_radius)s
        %(radius_step)s
        %(annulus_width)s
        %(extend_first_annulus)s
            In LRIC this specifically preserves juxtacrine (direct-contact)
            ligand-receptor signal in the first bin.
        %(cell_types)s
            Only relevant when ``groupby`` is set.
        %(min_cells)s
            Default ``None`` derives the threshold from slide composition
            instead of using a fixed count: cell types making up no more than
            1% of all cells are dropped.
        %(groupby_pairs)s
            Only relevant when ``groupby`` is set. Restricts the directed
            sender->receiver combinations actually computed to those listed;
            cell types referenced by ``groupby_pairs`` are also folded into
            ``cell_types``.
        %(expr_prop)s
            Computed within the relevant population: each cell type in
            pairwise mode, all cells in agnostic mode. Pairs below the
            threshold are set to ``NaN``.
        complex_sep
            Separator used to identify multi-subunit complexes in the resource
            (e.g. ``"_"`` splits ``"ITGAV_ITGB3"`` into its subunits and adds
            the minimum-subunit expression as a new column in ``adata.var``).
            Set to ``None`` to skip complex handling.
        %(lr_sep)s
        transform_fn
            Expression transform applied to ligand and receptor matrices,
            defaulting to mean-normalisation to 1 (:func:`_linear_transform`).
        %(use_raw)s
        %(layer)s
        pair_chunk
            Number of LR pairs processed per chunk when accumulating the
            weighted numerator; lower to reduce peak memory on very large
            resources.
        %(key_added)s
        %(inplace)s
        %(verbose)s

        Returns
        -------
        Result dict keys: cell_types, pair_names, radii, results
        """
        resource = _handle_resource(
            interactions=interactions,
            resource=resource,
            resource_name=resource_name,
            x_name="ligand",
            y_name="receptor",
            verbose=verbose,
        )

        if groupby is not None:
            groupby_pairs_subset = _get_groupby_subset(groupby_pairs)
            if groupby_pairs_subset is not None:
                cell_types = (
                    groupby_pairs_subset
                    if cell_types is None
                    else np.union1d(list(cell_types), groupby_pairs_subset)
                )

        _adata_orig = adata
        adata = prep_check_adata(
            adata=adata,
            groupby=groupby,
            min_cells=(
                _default_min_cells(adata, min_cells, verbose)
                if groupby is not None
                else None
            ),
            groupby_subset=cell_types if groupby is not None else None,
            use_raw=use_raw,
            layer=layer,
            obsm={spatial_key: adata.obsm[spatial_key]},
            complex_sep=complex_sep,
            verbose=verbose,
        )

        if complex_sep is not None:
            entities = np.union1d(resource["ligand"].astype(str), resource["receptor"].astype(str))
            if any(complex_sep in e for e in entities):
                adata = _add_complexes_to_var(adata, entities, complex_sep=complex_sep)

        resource = resource[
            resource["ligand"].isin(adata.var_names) & resource["receptor"].isin(adata.var_names)
        ]
        assert_covered(
            np.union1d(resource["ligand"], resource["receptor"]), adata.var_names, verbose=verbose
        )

        if groupby is None:
            res = self._agnostic(
                adata=adata,
                resource=resource,
                spatial_key=spatial_key,
                max_radius=max_radius,
                radius_step=radius_step,
                annulus_width=annulus_width,
                extend_first_annulus=extend_first_annulus,
                expr_prop=expr_prop,
                lr_sep=lr_sep,
                transform_fn=transform_fn,
                pair_chunk=pair_chunk,
                verbose=verbose,
            )
        else:
            res = self._pairwise(
                adata=adata,
                resource=resource,
                groupby=groupby,
                spatial_key=spatial_key,
                max_radius=max_radius,
                radius_step=radius_step,
                annulus_width=annulus_width,
                extend_first_annulus=extend_first_annulus,
                groupby_pairs=groupby_pairs,
                expr_prop=expr_prop,
                lr_sep=lr_sep,
                transform_fn=transform_fn,
                pair_chunk=pair_chunk,
                verbose=verbose,
            )

        if inplace:
            _adata_orig.uns[key_added] = res
            return None
        return res

    def _agnostic(
        self,
        adata: AnnData,
        resource: pd.DataFrame,
        spatial_key: str,
        max_radius: float,
        radius_step: float,
        annulus_width: float,
        extend_first_annulus: bool,
        expr_prop: float,
        lr_sep: str,
        transform_fn: Callable | None,
        pair_chunk: int,
        verbose: bool,
    ) -> dict:
        """Cell-type-agnostic LRIC across all cells (self-pairs excluded).

        Null: jointly permute which position gets which (ligand, receptor)
        weight *pair* -- both weights move together as a unit, since they
        belong to one cell. For a fixed edge (i, j) and random permutation
        pi, ``E_pi[wL(pi(i)) wR(pi(j))] = (S_L*S_R - sum_i wL(i)wR(i)) / (N(N-1))``,
        the average weighted product over all ``N(N-1)`` distinct ordered
        pairs; the self-pair correction term ``sum wL*wR`` accounts for every
        cell being able to pair with itself in the ``S_L*S_R`` product, while
        actual self-pairs are excluded from the edge set. Reduces exactly to
        ``CrossPCF``'s directed curve when weights are one-hot type
        indicators.

        """
        transform = _linear_transform if transform_fn is None else transform_fn
        _logg("Running cell-type-agnostic LRIC.", verbose=verbose)

        lr_pairs, unique_ligands, unique_receptors, pair_names = _index_resource(adata, resource, lr_sep)
        if lr_pairs.size == 0:
            raise ValueError("No LR pairs found in adata.var_names after filtering the resource.")

        coords = np.asarray(adata.obsm[spatial_key], dtype=float)
        N = len(coords)
        radii_inner, radii_outer = _make_radii(
            max_radius, radius_step, annulus_width, extend_first_annulus
        )
        n_bins = len(radii_inner)

        WL = _pair_weights(adata, unique_ligands, lr_pairs[:, 0], transform)
        WR = _pair_weights(adata, unique_receptors, lr_pairs[:, 1], transform)

        tree = cKDTree(coords)
        T = _bin_pair_counts(tree, tree, radii_inner, radii_outer)  # (n_bins,) whole-tissue support
        I, J, bin_idx = _support_edge_list(tree, radii_inner, radii_outer)
        num = _weighted_support_numerator(I, J, bin_idx, WL, WR, n_bins, N, pair_chunk)

        # closed-form null mean: T(b) * E[wL(i) wR(j)] over random distinct pairs
        S_L, S_R, cross = WL.sum(0), WR.sum(0), (WL * WR).sum(0)
        denom = T[:, None] * ((S_L * S_R - cross) / (N * (N - 1)))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = num / denom
            g[denom == 0] = np.nan
        lric = g.astype(np.float32)

        if expr_prop > 0:
            lric = _expr_prop_mask(
                lric, (WL > 0).sum(axis=0) / N, (WR > 0).sum(axis=0) / N, expr_prop
            )
        return {"pair_names": pair_names, "radii": radii_inner, "lric": lric}

    def _pairwise(
        self,
        adata: AnnData,
        resource: pd.DataFrame,
        groupby: str,
        spatial_key: str,
        max_radius: float,
        radius_step: float,
        annulus_width: float,
        extend_first_annulus: bool,
        groupby_pairs: pd.DataFrame | None,
        expr_prop: float,
        lr_sep: str,
        transform_fn: Callable | None,
        pair_chunk: int,
        verbose: bool,
    ) -> dict:
        """Cell-type pairwise ("ct") LRIC under the conditional (within-type) null.

        Weights are computed TISSUE-WIDE (all N cells). Null, in two stages:
        (1) roles are assigned across the tissue, giving
        ``E[T_SR(b)] = n_S n_R / (N(N-1)) * T_all(b)``; (2) each cell's
        ``(wL, wR)`` tuple is permuted only among cells of its own type, giving
        ``E[Num_SR(b) | T_SR(b)] = T_SR(b) * w̄L^S * w̄R^R``. Composed::

            E[Num_SR(b)] = n_S n_R / (N(N-1)) * T_all(b) * w̄L^S * w̄R^R

        Because ``w̄L^S`` / ``w̄R^R`` come from the type's own cells, a
        type-level marginal expression shift cancels exactly.

        ``T_SR(b)`` is read straight off the group bounds (edges per group),
        so no per-type KD-tree is needed.

        Reduces exactly to ``CrossPCF`` whenever weights are
        position-independent.
        """
        transform = _linear_transform if transform_fn is None else transform_fn

        obs_types = adata.obs[groupby].astype(str).values
        cell_types_list = sorted(np.unique(obs_types).tolist())
        n_types = len(cell_types_list)
        pairs = [(s, r) for s in cell_types_list for r in cell_types_list if s != r]
        if groupby_pairs is not None:
            requested = set(zip(groupby_pairs[P.source], groupby_pairs[P.target], strict=True))
            pairs = [p for p in pairs if p in requested]
        _logg(
            f"Running LRIC (conditional within-type null) for {n_types} "
            f"cell types ({len(pairs)} directed pairs).",
            verbose=verbose,
        )

        lr_pairs, unique_ligands, unique_receptors, pair_names = _index_resource(adata, resource, lr_sep)
        if lr_pairs.size == 0:
            raise ValueError("No LR pairs found in adata.var_names after filtering the resource.")

        coords = np.asarray(adata.obsm[spatial_key], dtype=float)
        N = len(coords)
        radii_inner, radii_outer = _make_radii(
            max_radius, radius_step, annulus_width, extend_first_annulus
        )
        n_bins = len(radii_inner)

        WL = _pair_weights(adata, unique_ligands, lr_pairs[:, 0], transform)
        WR = _pair_weights(adata, unique_receptors, lr_pairs[:, 1], transform)

        # per-type marginals: the heart of the conditional null
        mL = _type_mean_weights(WL, obs_types, cell_types_list)
        mR = _type_mean_weights(WR, obs_types, cell_types_list)

        counts = {ct: int((obs_types == ct).sum()) for ct in cell_types_list}

        tree = cKDTree(coords)
        T_all = _bin_pair_counts(tree, tree, radii_inner, radii_outer)  # geometry only
        I, J, bin_idx = _support_edge_list(tree, radii_inner, radii_outer)

        # Group the edge list by (sender type, receiver type, bin) with a single
        # stable sort, so that each directed pair's edges occupy a contiguous
        # slice addressable via `bounds`. On a large slide (big `max_radius` =>
        # tens of millions of edges) the edge list dominates memory, so the key
        # is int32 (keys are bounded by n_types^2 * n_bins, far below int32
        # range), is built in place, and every temporary is released as soon as
        # it has been consumed.
        type_code = pd.Categorical(obs_types, categories=cell_types_list).codes.astype(np.int32)
        n_groups = n_types * n_types * n_bins
        group_key = type_code[I].astype(np.int32)
        group_key *= n_types
        group_key += type_code[J]
        group_key *= n_bins
        np.add(group_key, bin_idx, out=group_key, casting="unsafe")
        order = np.argsort(group_key, kind="stable")
        bounds = _edge_group_bounds(group_key[order], n_groups)
        del group_key, bin_idx
        I_sorted = I[order]
        del I
        J_sorted = J[order]
        del J, order

        # Per-type expressing proportions, computed once (O(N * n_pairs)) and
        # reused by every directed pair involving that type.
        pexp_L = pexp_R = None
        if expr_prop > 0:
            n_lr = WL.shape[1]
            pexp_L = np.empty((n_types, n_lr), dtype=np.float64)
            pexp_R = np.empty((n_types, n_lr), dtype=np.float64)
            for ci, ct in enumerate(cell_types_list):
                ct_mask = obs_types == ct
                pexp_L[ci] = (WL[ct_mask] > 0).sum(axis=0) / counts[ct]
                pexp_R[ci] = (WR[ct_mask] > 0).sum(axis=0) / counts[ct]

        idx_of = {ct: i for i, ct in enumerate(cell_types_list)}
        results: dict[tuple[str, str], np.ndarray] = {}
        g_expr_d: dict[tuple[str, str], np.ndarray] = {}
        g_pcf_d: dict[tuple[str, str], np.ndarray] = {}

        for sender, receiver in tqdm(pairs, disable=not verbose, desc="LRIC"):
            si, ri = idx_of[sender], idx_of[receiver]
            n_S, n_R = counts[sender], counts[receiver]

            g0 = (si * n_types + ri) * n_bins
            grp = bounds[g0 : g0 + n_bins + 1]
            Num_SR = _segment_weighted_sums(I_sorted, J_sorted, grp, WL, WR, pair_chunk)
            # edges per group == observed ordered S->R pair count per bin
            T_SR = np.diff(grp).astype(np.float64)

            pair_prod = _null_pair_product(si, ri, mL, mR)
            exp_T = n_S * n_R / (N * (N - 1)) * T_all              # null pair count
            expected = exp_T[:, None] * pair_prod[None, :]         # null weighted pair-sum

            with np.errstate(divide="ignore", invalid="ignore"):
                g_full = Num_SR / expected
                g_full[expected == 0] = np.nan
                g_pcf = T_SR / exp_T
                g_pcf[exp_T == 0] = np.nan
                g_expr = Num_SR / (T_SR[:, None] * pair_prod[None, :])
                g_expr[(T_SR[:, None] * pair_prod[None, :]) == 0] = np.nan

            mat = g_full.astype(np.float32)
            mat_e = g_expr.astype(np.float32)
            if expr_prop > 0:
                mat = _expr_prop_mask(mat, pexp_L[si], pexp_R[ri], expr_prop)
                mat_e = _expr_prop_mask(mat_e, pexp_L[si], pexp_R[ri], expr_prop)

            results[(sender, receiver)] = mat
            g_expr_d[(sender, receiver)] = mat_e
            g_pcf_d[(sender, receiver)] = g_pcf

        out = {
            "cell_types": cell_types_list,
            "pair_names": pair_names,
            "radii": radii_inner,
            "results": results,  # (n_bins, n_pairs) per pair. Architecture x expression coupling
            "g_expr": g_expr_d, # (n_bins, n_pairs) per pair -- expression coupling ALONE, conditioned on where
            # the cells already sit.
            "g_pcf": g_pcf_d,  # (n_bins,) per pair -- architecture ALONE, equals cross_pcf
        }
        return out

lric = LRIC()