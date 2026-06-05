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
from liana.resource.select_resource import _handle_resource


# ── helpers ───────────────────────────────────────────────────────────

def _linear_transform(expr: np.ndarray) -> np.ndarray:
    """Mean-normalise expression to a mean of 1, clipped at 0."""
    mean = expr.mean(axis=0, keepdims=True)
    return np.maximum(expr / (mean + 1e-12), 0)

#Do we need this or just do it inline?
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


#Do you suggest that I use prep_check_adata (existing util) here? I prefer to keep this as we only need to filter cts not validate anything else.

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
    max_radius: float, radius_step: float, annulus_width: float
) -> tuple[np.ndarray, np.ndarray]:
    radii_inner = np.arange(radius_step, max_radius + radius_step, radius_step, dtype=float)
    return radii_inner, radii_inner + annulus_width


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (receiver, sender) index pairs within ``max_r`` with pairwise distances."""
    tree = cKDTree(send_coords)
    neigh_lists = tree.query_ball_point(recv_coords, max_r)

    rows, cols = [], []
    for r_i, neigh in enumerate(neigh_lists):
        neigh_arr = np.asarray(neigh, dtype=np.int32)
        if exclude_self:
            neigh_arr = neigh_arr[neigh_arr != r_i]
        if neigh_arr.size == 0:
            continue
        rows.append(np.full(neigh_arr.size, r_i, dtype=np.int32))
        cols.append(neigh_arr)

    if not rows:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    dists = np.linalg.norm(
        recv_coords[all_rows] - send_coords[all_cols], axis=1
    ).astype(np.float32)
    return all_rows, all_cols, dists


def _wpcf_bins(
    sender_w: np.ndarray,
    receiver_w: np.ndarray,
    corrected_areas: np.ndarray,
    sender_density: float,
    all_rows: np.ndarray,
    all_cols: np.ndarray,
    rel_dists: np.ndarray,
    nRcv: int,
    nSnd: int,
    radii_inner: np.ndarray,
    radii_outer: np.ndarray,
) -> np.ndarray:
    """Core wPCF bin accumulation. Returns ``(n_bins, n_pairs)`` float32."""
    n_bins, n_pairs = radii_inner.size, sender_w.shape[1]
    out = np.zeros((n_bins, n_pairs), dtype=np.float32)

    for b, (r_in, r_out) in enumerate(zip(radii_inner, radii_outer)):
        expected_b = sender_density * corrected_areas[:, b]
        valid = np.isfinite(expected_b)
        if not valid.any() or rel_dists.size == 0:
            continue
        m = (rel_dists >= r_in) & (rel_dists < r_out)
        if not m.any():
            continue

        adj = sparse.csr_matrix(
            (np.ones(m.sum(), dtype=np.float32), (all_rows[m], all_cols[m])),
            shape=(nRcv, nSnd),
        )
        lig_counts = adj @ sender_w
        safe_exp = np.where(valid, expected_b, 1.0)[:, None]
        contrib = np.where(valid[:, None], lig_counts / safe_exp, 0.0)
        recv_w_valid = receiver_w * valid[:, None]
        num = (recv_w_valid * contrib).sum(axis=0)
        denom = recv_w_valid.sum(axis=0)
        out[b] = np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0).astype(
            np.float32
        )

    return out


# ── CrossPCF ──────────────────────────────────────────────────────────────────


class CrossPCF:
    """Cross pair-correlation function (cross-PCF) between cell types."""

    @d.dedent
    def __call__(
        self,
        adata: AnnData,
        groupby: str,
        spatial_key: str = K.spatial_key,
        cell_types: Iterable[str] | None = None,
        min_cells: int = V.min_cells,
        max_radius: float = 300,
        radius_step: float = 20,
        annulus_width: float = 20,
        n_angle_samples: int = 360,
        key_added: str = "cross_pcf",
        inplace: bool = V.inplace,
        verbose: bool = V.verbose,
    ) -> dict | None:
        """
        Cross pair-correlation function for all directed cell-type pairs.

        Computes the distance-resolved cross-PCF for every directed
        sender→receiver combination of cell types present in
        ``adata.obs[groupby]``.

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
            Outer edge of the last annulus.
        radius_step
            Step between successive annulus inner edges.
        annulus_width
            Ring width of each annulus.
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

        radii_inner, radii_outer = _make_radii(max_radius, radius_step, annulus_width)
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


# ── WeightedPCF ───────────────────────────────────────────────────────────────


class WeightedPCF:
    """LR-weighted cross pair-correlation function (wPCF)."""

    @d.dedent
    def __call__(
        self,
        adata: AnnData,
        resource: pd.DataFrame | None = V.resource,
        resource_name: str | None = None,
        interactions: list | None = V.interactions,
        groupby: str | None = None,
        spatial_key: str = K.spatial_key,
        max_radius: float = 300,
        radius_step: float = 20,
        annulus_width: float = 20,
        cell_types: Iterable[str] | None = None,
        min_cells: int = V.min_cells,
        n_angle_samples: int = 360,
        transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        use_raw: bool = V.use_raw,
        layer: str | None = V.layer,
        key_added: str = "lr_wpcf",
        inplace: bool = V.inplace,
        verbose: bool = V.verbose,
    ) -> dict | None:
        """
        LR-weighted cross pair-correlation function (wPCF).

        When ``groupby`` is ``None`` (default), all cells are treated as
        potential senders and receivers (self-pairs excluded), providing a
        global screen for LR pairs with strong spatial co-enrichment signal.
        Result dict keys: ``pair_names``, ``radii``, ``wpcf``
        (shape ``(n_bins, n_pairs)``).

        When ``groupby`` is a column name in ``adata.obs``, the wPCF is
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
            Outer edge of the last annulus.
        radius_step
            Step between successive annulus inner edges.
        annulus_width
            Ring width of each annulus.
        cell_types
            Subset of cell types to consider (only when ``groupby`` is set).
            Defaults to all types in ``adata.obs[groupby]``.
        %(min_cells)s
        n_angle_samples
            Angular resolution for bounding-box edge correction
            (360 ≈ 0.3 %% error).
        transform_fn
            Expression transform applied to ligand and receptor matrices.
            Defaults to mean→1 normalization (:func:`_linear_transform`).
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

        if groupby is None:
            res = self._agnostic(
                adata=adata,
                resource=resource,
                spatial_key=spatial_key,
                max_radius=max_radius,
                radius_step=radius_step,
                annulus_width=annulus_width,
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
                cell_types=cell_types,
                min_cells=min_cells,
                n_angle_samples=n_angle_samples,
                transform_fn=transform_fn,
                use_raw=use_raw,
                layer=layer,
                verbose=verbose,
            )

        if inplace:
            adata.uns[key_added] = res
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
        n_angle_samples: int,
        transform_fn: Callable | None,
        use_raw: bool,
        layer: str | None,
        verbose: bool,
    ) -> dict:
        """Cell-type-agnostic wPCF across all cells (self-pairs excluded)."""
        transform = _linear_transform if transform_fn is None else transform_fn

        coords = np.asarray(adata.obsm[spatial_key], dtype=float)
        n_cells = coords.shape[0]

        _logg("Running cell-type-agnostic LR-wPCF.", verbose=verbose)

        lr_pairs, unique_ligands, unique_receptors, pair_names = _index_resource(adata, resource)
        if lr_pairs.size == 0:
            raise ValueError(
                "No LR pairs found in adata.var_names after filtering the resource."
            )

        all_mask = np.ones(n_cells, dtype=bool)
        sender_w = transform(
            _get_expr(adata, all_mask, unique_ligands, use_raw, layer)
        )[:, lr_pairs[:, 0]]
        receiver_w = transform(
            _get_expr(adata, all_mask, unique_receptors, use_raw, layer)
        )[:, lr_pairs[:, 1]]

        radii_inner, radii_outer = _make_radii(max_radius, radius_step, annulus_width)
        corrected_areas, x_min, x_max, y_min, y_max = _corrected_areas(
            coords, coords, radii_inner, radii_outer, n_angle_samples
        )
        sender_density = (n_cells - 1) / ((x_max - x_min) * (y_max - y_min))

        all_rows, all_cols, rel_dists = _spatial_pairs(
            coords, coords, float(radii_outer[-1]), exclude_self=True
        )
        wpcf = _wpcf_bins(
            sender_w, receiver_w, corrected_areas, sender_density,
            all_rows, all_cols, rel_dists, n_cells, n_cells, radii_inner, radii_outer,
        )
        return {"pair_names": pair_names, "radii": radii_inner, "wpcf": wpcf}

    def _pairwise(
        self,
        adata: AnnData,
        resource: pd.DataFrame,
        groupby: str,
        spatial_key: str,
        max_radius: float,
        radius_step: float,
        annulus_width: float,
        cell_types: Iterable[str] | None,
        min_cells: int,
        n_angle_samples: int,
        transform_fn: Callable | None,
        use_raw: bool,
        layer: str | None,
        verbose: bool,
    ) -> dict:
        """Cell-type-specific LR-wPCF for all directed sender→receiver pairs."""
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
            f"Running LR-wPCF for {len(cell_types_list)} cell types "
            f"({len(pairs)} directed pairs).",
            verbose=verbose,
        )

        lr_pairs, unique_ligands, unique_receptors, pair_names = _index_resource(adata, resource)
        if lr_pairs.size == 0:
            raise ValueError(
                "No LR pairs found in adata.var_names after filtering the resource."
            )

        radii_inner, radii_outer = _make_radii(max_radius, radius_step, annulus_width)
        results: dict[tuple[str, str], np.ndarray] = {}

        for sender, receiver in tqdm(pairs, disable=not verbose, desc="LR-wPCF"):
            recv_mask = obs_types == receiver
            send_mask = obs_types == sender
            recv_coords = np.asarray(adata.obsm[spatial_key][recv_mask], dtype=float)
            send_coords = np.asarray(adata.obsm[spatial_key][send_mask], dtype=float)
            nRcv, nSnd = recv_coords.shape[0], send_coords.shape[0]

            sender_w = transform(
                _get_expr(adata, send_mask, unique_ligands, use_raw, layer)
            )[:, lr_pairs[:, 0]]
            receiver_w = transform(
                _get_expr(adata, recv_mask, unique_receptors, use_raw, layer)
            )[:, lr_pairs[:, 1]]

            corrected_areas, x_min, x_max, y_min, y_max = _corrected_areas(
                recv_coords,
                np.vstack([recv_coords, send_coords]),
                radii_inner,
                radii_outer,
                n_angle_samples,
            )
            sender_density = nSnd / ((x_max - x_min) * (y_max - y_min))

            all_rows, all_cols, rel_dists = _spatial_pairs(
                recv_coords, send_coords, float(radii_outer[-1])
            )
            wpcf = _wpcf_bins(
                sender_w, receiver_w, corrected_areas, sender_density,
                all_rows, all_cols, rel_dists, nRcv, nSnd, radii_inner, radii_outer,
            )
            results[(sender, receiver)] = wpcf

        return {
            "cell_types": cell_types_list,
            "pair_names": pair_names,
            "radii": radii_inner,
            "results": results,
        }


lr_wpcf = WeightedPCF()
