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
from liana._docs import d
from liana._logging import _logg
from liana.method.sp._utils import _add_complexes_to_var
from liana.resource.select_resource import _handle_resource

# ── helpers ───────────────────────────────────────────────────────────

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
    cell_mask: np.ndarray,
    gene_names: list[str],
    use_raw: bool = False,
    layer: str | None = None,
) -> np.ndarray:
    sub = adata[cell_mask][:, gene_names]
    if use_raw:
        if sub.raw is None:
            raise AttributeError("`.raw` is None — set `use_raw=False`.")
        return _to_dense(sub.raw.X)
    if layer is not None:
        return _to_dense(sub.layers[layer])
    return _to_dense(sub.X)


def _pair_weights(
    adata: AnnData,
    cell_mask: np.ndarray,
    gene_names: list[str],
    idx: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    use_raw: bool = False,
    layer: str | None = None,
) -> np.ndarray:
    """Per-pair expression weights.

    Transform the unique-gene matrix, then gather to per-pair columns ``idx``;
    returns ``(n_cells_in_mask, n_pairs)``.
    """
    return transform(_get_expr(adata, cell_mask, gene_names, use_raw, layer))[:, idx]


def _index_resource(
    adata: AnnData,
    resource: pd.DataFrame,
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
        resource_f["ligand"].astype(str) + V.lr_sep + resource_f["receptor"].astype(str)
    ).tolist()

    return lr_pairs, unique_ligands, unique_receptors, pair_names


def _filter_by_min_cells(
    obs_types: np.ndarray,
    cell_types_list: list[str],
    min_cells: int,
    verbose: bool = V.verbose,
) -> list[str]:
    """Drop cell types whose count falls below ``min_cells`` and warn."""
    counts = {ct: int((obs_types == ct).sum()) for ct in cell_types_list}
    kept = [ct for ct, n in counts.items() if n >= min_cells]
    dropped = [ct for ct, n in counts.items() if n < min_cells]
    if dropped:
        _logg(
            f"Dropping {len(dropped)} cell type(s) with fewer than {min_cells} cells: {dropped}",
            level="warn",
            verbose=verbose,
        )
    return kept


def _make_radii(
    max_radius: float, radius_step: float, annulus_width: float, extend_first_annulus: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    radii_inner = np.arange(radius_step, max_radius + radius_step, radius_step, dtype=float)
    radii_outer = radii_inner + annulus_width
    if extend_first_annulus:
        radii_inner[0] = 0.0  # merge the [0, radius_step) contact band into the first annulus
    return radii_inner, radii_outer


def _circle_bbox_fractions(
    centers: np.ndarray,
    radii: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_samples: int = 360,
) -> np.ndarray:
    """
    Fraction of each circle's circumference that lies inside the bounding box.

    Parameters
    ----------
    centers : (n_points, 2)
    radii : (n_radii,)

    Returns
    -------
    fractions : (n_points, n_radii) in ``(0, 1]``
    """
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    px = centers[:, 0, None, None] + radii[None, :, None] * np.cos(angles)[None, None, :]
    py = centers[:, 1, None, None] + radii[None, :, None] * np.sin(angles)[None, None, :]
    inside = (px >= x_min) & (px <= x_max) & (py >= y_min) & (py <= y_max)
    return inside.mean(axis=-1)


def _corrected_areas(
    recv_coords: np.ndarray,
    all_coords: np.ndarray,
    radii_inner: np.ndarray,
    radii_outer: np.ndarray,
    n_angle_samples: int,
) -> tuple[np.ndarray, float, float, float, float]:
    """Bounding-box-corrected annulus areas for each receiver position."""
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    fracs_outer = _circle_bbox_fractions(
        recv_coords, radii_outer, x_min, x_max, y_min, y_max, n_angle_samples
    )
    fracs_inner = _circle_bbox_fractions(
        recv_coords, radii_inner, x_min, x_max, y_min, y_max, n_angle_samples
    )
    areas = np.pi * radii_outer**2 * fracs_outer - np.pi * radii_inner**2 * fracs_inner
    return np.where(areas > 0, areas, np.nan), x_min, x_max, y_min, y_max


def _spatial_pairs(
    recv_coords: np.ndarray,
    send_coords: np.ndarray,
    max_r: float,
    exclude_self: bool = False,
    tree: cKDTree | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (receiver, sender) index pairs within ``max_r`` with pairwise distances."""
    if tree is None:
        tree = cKDTree(send_coords)
    neigh_lists = tree.query_ball_point(recv_coords, max_r, workers=-1)

    lengths = np.fromiter((len(n) for n in neigh_lists), dtype=np.int32, count=len(recv_coords))
    if lengths.sum() == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)

    all_rows = np.repeat(np.arange(len(recv_coords), dtype=np.int32), lengths)
    all_cols = np.concatenate([np.asarray(n, dtype=np.int32) for n in neigh_lists if n])

    if exclude_self:
        mask = all_rows != all_cols
        all_rows, all_cols = all_rows[mask], all_cols[mask]

    if all_rows.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)

    dists = np.linalg.norm(
        recv_coords[all_rows] - send_coords[all_cols], axis=1
    ).astype(np.float32)
    return all_rows, all_cols, dists


def _lric_bins(
    sender_w: np.ndarray,
    receiver_w: np.ndarray,
    corrected_areas: np.ndarray,
    sender_density: float,
    all_rows: np.ndarray,
    all_cols: np.ndarray,
    rel_dists: np.ndarray,
    radii_inner: np.ndarray,
    radii_outer: np.ndarray,
) -> np.ndarray:
    """Core LRIC bin accumulation. Returns ``(n_bins, n_pairs)`` float32."""
    expected = (sender_density * corrected_areas).astype(np.float32)  # (n_recv, n_bins)
    valid = np.isfinite(expected)                                      # (n_recv, n_bins)
    safe_exp = np.where(valid, expected, 1.0).astype(np.float32)

    # denom[b, p] = sum_i valid[i,b] * receiver_w[i, p]
    denom = valid.astype(np.float32).T @ receiver_w                   # (n_bins, n_pairs)

    # scale[k, b] = 1/expected[i,b]  if edge k=(i,j) is in annulus b and valid, else 0
    pair_bins = (rel_dists[:, None] >= radii_inner) & (rel_dists[:, None] < radii_outer)
    scale = (pair_bins & valid[all_rows]).astype(np.float32) / safe_exp[all_rows]  # (n_edges, n_bins)

    # numerator[b, p] = sum_{k in bin b, valid} receiver_w[i,p] * sender_w[j,p] / expected[i,b]
    edge_cross_w = receiver_w[all_rows] * sender_w[all_cols]          # (n_edges, n_pairs)
    numerator = scale.T @ edge_cross_w                                 # (n_bins, n_pairs)

    # bins with no valid receiver weight (denom == 0) stay 0; avoids 0/0.
    out = np.zeros_like(numerator)
    np.divide(numerator, denom, out=out, where=denom > 0)
    return out.astype(np.float32)


def _min_expressing_mask(
    mat: np.ndarray, n_snd: np.ndarray, n_rcv: np.ndarray, min_expressing: int
) -> np.ndarray:
    """NaN out pairs whose sender or receiver expressing-cell count is below the threshold."""
    if min_expressing <= 0:
        return mat
    filt = (n_snd < min_expressing) | (n_rcv < min_expressing)
    if filt.any():
        mat = mat.copy()
        mat[:, filt] = np.nan
    return mat


# ── CrossPCF ──────────────────────────────────────────────────────────────────


class CrossPCF:
    """Cross pair-correlation function (cross-PCF) between cell types.

    The cross-PCF, ``g(r)``, measures how the density of receiver-type cells
    at distance ``r`` from a sender-type cell compares to the density expected
    under complete spatial randomness. ``g(r) > 1`` means the two cell types
    co-localise at that distance, ``g(r) < 1`` means they avoid one another,
    and ``g(r) ≈ 1`` means they are placed independently. It is used to map
    tissue architecture — the length scales at which cell types cluster or
    segregate — using cell positions alone, ignoring gene expression.

    The cross-PCF is a classical point-pattern spatial statistic; this
    implementation was inspired by the cross-PCF in the MuSpAn toolbox and
    their applications in the accompanying manuscript (Bull et al., 2024, doi:10.1101/2024.12.06.627195).
    """

    @d.dedent
    def __call__(
        self,
        adata: AnnData,
        groupby: str,
        spatial_key: str = K.spatial_key,
        cell_types: Iterable[str] | None = None,
        min_cells: int = V.min_cells,
        max_radius: float = 200,
        radius_step: float = 20,
        annulus_width: float = 20,
        extend_first_annulus: bool = True,
        n_angle_samples: int = 360,
        key_added: str = "cross_pcf",
        inplace: bool = V.inplace,
        verbose: bool = V.verbose,
    ) -> dict | None:
        """
        Cross pair-correlation function for all directed cell-type pairs.

        Computes the distance-resolved cross-PCF ``g(r)`` for every directed
        sender→receiver combination of cell types present in
        ``adata.obs[groupby]``. ``g(r)`` compares the observed density of
        receiver cells at distance ``r`` from each sender to the density
        expected under spatial randomness, revealing the length scales at
        which cell types co-localise (``> 1``) or avoid one another (``< 1``).
        Expression is not used — this characterises tissue architecture alone.

        Parameters
        ----------
        %(adata)s
        %(groupby)s
        %(spatial_key)s
        cell_types
            Subset of cell types to consider. Defaults to all types in
            ``adata.obs[groupby]``.
        %(min_cells)s
        max_radius
            Inner edge of the last (widest) annulus bin; the outer edge extends
            to ``max_radius + annulus_width``.
        radius_step
            Step between successive annulus inner edges.
        annulus_width
            Ring width of each annulus.
        extend_first_annulus
            If ``True`` (default), extend the first annulus inward to start at
            radius 0 (spanning ``[0, radius_step + annulus_width)``) rather than
            at ``radius_step``. Cell centroids cannot lie closer than ~one cell
            diameter, so the innermost band is otherwise a thin, near-empty,
            high-variance bin; extending it folds genuine cell-cell contact
            pairs into the first bin instead of discarding them. ``False`` keeps
            the first annulus at ``[radius_step, radius_step + annulus_width)``.
        n_angle_samples
            Angular resolution for bounding-box edge correction
            (360 ≈ 0.3 %% error).
        %(key_added)s
        %(inplace)s
        %(verbose)s

        Returns
        -------
        dict with keys ``cell_types``, ``radii``, ``results`` if
        ``inplace=False``, else ``None``.
        ``results`` maps ``(sender, receiver)`` tuples to ``(n_bins,)`` arrays.
        """
        obs_types = adata.obs[groupby].astype(str).values
        cell_types_list = (
            sorted(np.unique(obs_types).tolist())
            if cell_types is None
            else [str(c) for c in cell_types]
        )
        cell_types_list = _filter_by_min_cells(obs_types, cell_types_list, min_cells, verbose)
        pairs = [(s, r) for s in cell_types_list for r in cell_types_list if s != r]

        _logg(
            f"Computing cross-PCF for {len(cell_types_list)} cell types "
            f"({len(pairs)} directed pairs).",
            verbose=verbose,
        )

        results: dict[tuple[str, str], np.ndarray] = {}
        radii_ref: np.ndarray | None = None

        for sender, receiver in tqdm(pairs, disable=not verbose, desc="cross-PCF"):
            pcf_vals, radii = self._compute_pair(
                adata=adata,
                receiver_cell_type=receiver,
                sender_cell_type=sender,
                groupby=groupby,
                spatial_key=spatial_key,
                max_radius=max_radius,
                radius_step=radius_step,
                annulus_width=annulus_width,
                extend_first_annulus=extend_first_annulus,
                n_angle_samples=n_angle_samples,
            )
            if radii_ref is None:
                radii_ref = radii
            results[(sender, receiver)] = pcf_vals

        res = {"cell_types": cell_types_list, "radii": radii_ref, "results": results}
        if inplace:
            adata.uns[key_added] = res
            return None
        return res

    def _compute_pair(
        self,
        adata: AnnData,
        receiver_cell_type: str,
        sender_cell_type: str,
        groupby: str,
        spatial_key: str,
        max_radius: float,
        radius_step: float,
        annulus_width: float,
        extend_first_annulus: bool,
        n_angle_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cross-PCF for a single sender→receiver cell-type pair."""
        obs_ct = adata.obs[groupby].astype(str).values
        A = np.asarray(adata.obsm[spatial_key][obs_ct == receiver_cell_type], dtype=float)
        B = np.asarray(adata.obsm[spatial_key][obs_ct == sender_cell_type], dtype=float)

        if A.shape[0] == 0 or B.shape[0] == 0:
            raise ValueError(
                f"Empty population(s): receiver='{receiver_cell_type}' ({A.shape[0]}), "
                f"sender='{sender_cell_type}' ({B.shape[0]})."
            )

        radii_inner, radii_outer = _make_radii(
            max_radius, radius_step, annulus_width, extend_first_annulus
        )
        corrected_areas, x_min, x_max, y_min, y_max = _corrected_areas(
            A, np.vstack([A, B]), radii_inner, radii_outer, n_angle_samples
        )
        density_B = B.shape[0] / ((x_max - x_min) * (y_max - y_min))

        tree_B = cKDTree(B)
        outer_counts = np.array(
            [tree_B.query_ball_point(A, float(r), return_length=True) for r in radii_outer]
        ).T
        inner_counts = np.array(
            [tree_B.query_ball_point(A, float(r), return_length=True) for r in radii_inner]
        ).T
        annulus_counts = outer_counts - inner_counts

        return np.nanmean(annulus_counts / (density_B * corrected_areas), axis=0), radii_inner


cross_pcf = CrossPCF()


# ── LRIC ───────────────────────────────────────────────────────────────


class LRIC:
    """Ligand-Receptor Interaction Correlation (LRIC).

    LRIC is an expression-weighted cross pair-correlation function: each cell's
    contribution at distance ``r`` is weighted by its ligand (sender) and
    receptor (receiver) expression. The resulting ``g(r)`` therefore reflects
    whether ligand- and receptor-expressing cells are spatially co-enriched at
    distance ``r``, beyond what cell-type co-localisation alone would predict.
    ``g(r) > 1`` means LR-expressing cells cluster at that distance, ``≈ 1``
    means expression adds nothing over the underlying spatial structure, and
    ``< 1`` means they are depleted. This surfaces candidate ligand-receptor
    interactions that are both spatially proximal and co-expressed — a more
    specific signal of potential cell-cell communication than spatial
    co-localisation or co-expression considered alone.

    LRIC builds on the cross pair-correlation function (cross-PCF); see
    :class:`CrossPCF`.
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
        min_cells: int = V.min_cells,
        min_expressing: int = 0,
        n_angle_samples: int = 360,
        complex_sep: str | None = V.complex_sep,
        transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        use_raw: bool = V.use_raw,
        layer: str | None = V.layer,
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
        max_radius
            Inner edge of the last (widest) annulus bin; the outer edge extends
            to ``max_radius + annulus_width``.
        radius_step
            Step between successive annulus inner edges.
        annulus_width
            Ring width of each annulus.
        extend_first_annulus
            If ``True`` (default), extend the first annulus inward to start at
            radius 0 (spanning ``[0, radius_step + annulus_width)``) rather than
            at ``radius_step``. Cell centroids cannot lie closer than ~one cell
            diameter, so the innermost band is otherwise a thin, near-empty,
            high-variance bin; extending it folds genuine cell-cell contact
            (juxtacrine) pairs into the first bin instead of discarding them.
            ``False`` keeps the first annulus at ``[radius_step, radius_step +
            annulus_width)``.
        cell_types
            Subset of cell types to consider (only when ``groupby`` is set).
            Defaults to all types in ``adata.obs[groupby]``.
        %(min_cells)s
        min_expressing
            Minimum number of cells (within the relevant population: each cell∂
            type in pairwise mode, all cells in agnostic mode) that must express
            the ligand or receptor for an LR pair to be kept; pairs below the
            threshold are set to ``NaN``. Default ``0`` keeps all results.
        n_angle_samples
            Angular resolution for bounding-box edge correction
            (360 ≈ 0.3 %% error).
        complex_sep
            Separator used to identify multi-subunit complexes in the resource
            (e.g. ``"_"`` splits ``"ITGAV_ITGB3"`` into its subunits and adds
            the minimum-subunit expression as a new column in ``adata.var``).
            Set to ``None`` to skip complex handling.
        transform_fn
            Expression transform applied to ligand and receptor matrices,
            defaulting to mean→1 normalization (:func:`_linear_transform`). It is
            applied per population: globally in agnostic mode, within each cell
            type in pairwise mode.
        %(use_raw)s
        %(layer)s
        %(key_added)s
        %(inplace)s
        %(verbose)s

        Returns
        -------
        ``dict`` if ``inplace=False``, else ``None``.
        """
        resource = _handle_resource(
            interactions=interactions,
            resource=resource,
            resource_name=resource_name,
            x_name="ligand",
            y_name="receptor",
            verbose=verbose,
        )

        _adata_orig = adata
        if complex_sep is not None:
            entities = np.union1d(resource["ligand"].astype(str), resource["receptor"].astype(str))
            if any(complex_sep in e for e in entities):
                adata = _add_complexes_to_var(adata, entities, complex_sep=complex_sep)

        if groupby is None:
            res = self._agnostic(
                adata=adata,
                resource=resource,
                spatial_key=spatial_key,
                max_radius=max_radius,
                radius_step=radius_step,
                annulus_width=annulus_width,
                extend_first_annulus=extend_first_annulus,
                min_expressing=min_expressing,
                n_angle_samples=n_angle_samples,
                transform_fn=transform_fn,
                use_raw=use_raw,
                layer=layer,
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
                cell_types=cell_types,
                min_cells=min_cells,
                min_expressing=min_expressing,
                n_angle_samples=n_angle_samples,
                transform_fn=transform_fn,
                use_raw=use_raw,
                layer=layer,
                verbose=verbose,
            )

        if inplace:
            _adata_orig.uns[key_added] = res
            return None
        return res

    def _compute_pair(
        self,
        send_coords: np.ndarray,
        recv_coords: np.ndarray,
        all_coords: np.ndarray,
        sender_w: np.ndarray,
        receiver_w: np.ndarray,
        radii_inner: np.ndarray,
        radii_outer: np.ndarray,
        n_angle_samples: int,
        exclude_self: bool = False,
        tree: cKDTree | None = None,
    ) -> np.ndarray:
        """Core LRIC computation for one sender/receiver population pair."""
        corrected_areas, x_min, x_max, y_min, y_max = _corrected_areas(
            recv_coords, all_coords, radii_inner, radii_outer, n_angle_samples
        )
        density = (send_coords.shape[0] - int(exclude_self)) / (
            (x_max - x_min) * (y_max - y_min)
        )
        all_rows, all_cols, rel_dists = _spatial_pairs(
            recv_coords, send_coords, float(radii_outer[-1]), exclude_self=exclude_self, tree=tree,
        )
        return _lric_bins(
            sender_w, receiver_w, corrected_areas, density,
            all_rows, all_cols, rel_dists,
            radii_inner, radii_outer,
        )

    def _agnostic(
        self,
        adata: AnnData,
        resource: pd.DataFrame,
        spatial_key: str,
        max_radius: float,
        radius_step: float,
        annulus_width: float,
        extend_first_annulus: bool,
        min_expressing: int,
        n_angle_samples: int,
        transform_fn: Callable | None,
        use_raw: bool,
        layer: str | None,
        verbose: bool,
    ) -> dict:
        """Cell-type-agnostic LRIC across all cells (self-pairs excluded)."""
        transform = _linear_transform if transform_fn is None else transform_fn

        _logg("Running cell-type-agnostic LRIC.", verbose=verbose)

        lr_pairs, unique_ligands, unique_receptors, pair_names = _index_resource(adata, resource)
        if lr_pairs.size == 0:
            raise ValueError(
                "No LR pairs found in adata.var_names after filtering the resource."
            )

        coords = np.asarray(adata.obsm[spatial_key], dtype=float)
        all_mask = np.ones(coords.shape[0], dtype=bool)
        # weights normalised globally over all cells (one population)
        sender_w = _pair_weights(adata, all_mask, unique_ligands, lr_pairs[:, 0], transform, use_raw, layer)
        receiver_w = _pair_weights(adata, all_mask, unique_receptors, lr_pairs[:, 1], transform, use_raw, layer)

        radii_inner, radii_outer = _make_radii(
            max_radius, radius_step, annulus_width, extend_first_annulus
        )
        lric = self._compute_pair(
            coords, coords, coords, sender_w, receiver_w,
            radii_inner, radii_outer, n_angle_samples, exclude_self=True,
        )
        if min_expressing > 0:
            lric = _min_expressing_mask(
                lric, (sender_w > 0).sum(axis=0), (receiver_w > 0).sum(axis=0), min_expressing
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
        cell_types: Iterable[str] | None,
        min_cells: int,
        min_expressing: int,
        n_angle_samples: int,
        transform_fn: Callable | None,
        use_raw: bool,
        layer: str | None,
        verbose: bool,
    ) -> dict:
        """Cell-type-specific LRIC for all directed sender→receiver pairs."""
        transform = _linear_transform if transform_fn is None else transform_fn

        obs_types = adata.obs[groupby].astype(str).values
        cell_types_list = (
            sorted(np.unique(obs_types).tolist())
            if cell_types is None
            else [str(c) for c in cell_types]
        )
        cell_types_list = _filter_by_min_cells(obs_types, cell_types_list, min_cells, verbose)
        pairs = [(s, r) for s in cell_types_list for r in cell_types_list if s != r]

        _logg(
            f"Running LRIC for {len(cell_types_list)} cell types "
            f"({len(pairs)} directed pairs).",
            verbose=verbose,
        )

        lr_pairs, unique_ligands, unique_receptors, pair_names = _index_resource(adata, resource)
        if lr_pairs.size == 0:
            raise ValueError(
                "No LR pairs found in adata.var_names after filtering the resource."
            )

        radii_inner, radii_outer = _make_radii(
            max_radius, radius_step, annulus_width, extend_first_annulus
        )

        # per cell type: coords, weights (normalised within the type), KD-tree, expressing counts
        ct_cache = {}
        for ct in cell_types_list:
            mask = obs_types == ct
            coords = np.asarray(adata.obsm[spatial_key][mask], dtype=float)
            sw = _pair_weights(adata, mask, unique_ligands, lr_pairs[:, 0], transform, use_raw, layer)
            rw = _pair_weights(adata, mask, unique_receptors, lr_pairs[:, 1], transform, use_raw, layer)
            ct_cache[ct] = (coords, sw, rw, cKDTree(coords),
                            (sw > 0).sum(axis=0), (rw > 0).sum(axis=0))

        results: dict[tuple[str, str], np.ndarray] = {}
        for sender, receiver in tqdm(pairs, disable=not verbose, desc="LRIC"):
            send_coords, sender_w, _, send_tree, n_snd, _ = ct_cache[sender]
            recv_coords, _, receiver_w, _, _, n_rcv = ct_cache[receiver]
            mat = self._compute_pair(
                send_coords, recv_coords, np.vstack([recv_coords, send_coords]),
                sender_w, receiver_w, radii_inner, radii_outer, n_angle_samples,
                tree=send_tree,
            )
            results[(sender, receiver)] = _min_expressing_mask(mat, n_snd, n_rcv, min_expressing)

        return {
            "cell_types": cell_types_list,
            "pair_names": pair_names,
            "radii": radii_inner,
            "results": results,
        }


lric = LRIC()
