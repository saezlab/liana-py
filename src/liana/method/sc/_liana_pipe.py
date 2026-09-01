from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import pandas
import pandas as pd
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix
from scipy.stats import norm

from liana._core._constants import CommonColumns as C
from liana._core._constants import DeMethod
from liana._core._constants import InternalValues as I
from liana._core._constants import MethodColumns as M
from liana._core._constants import PrimaryColumns as P
from liana._core._docs import d
from liana._core._pipe_utils import assert_covered, filter_resource, prep_check_adata
from liana._core._pipe_utils._aggregate import _aggregate
from liana._core._pipe_utils._common import _get_groupby_subset, _get_props, _join_stats
from liana._core._pipe_utils._get_mean_perms import _AggFn, _get_mat_idx, _get_means_perms
from liana._core._pipe_utils._pre import _choose_mtx_rep
from liana._core._types import get_obs, get_x
from liana.multisample.mdata_to_anndata import mdata_to_anndata
from liana.preprocessing.spatial_neighbors import spatial_pair_proximity
from liana.resource._reassemble_complexes import _explode_complexes, _filter_reassemble_complexes
from liana.resource.select_resource import _handle_resource

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from liana._core._types import MatrixLike
    from liana.method.sc._Method import MethodMeta


class MdataKwargs(TypedDict, total=False):
    """The `mdata_to_anndata` parameters `liana_pipe` forwards."""

    x_mod: str
    y_mod: str
    x_layer: str | None
    y_layer: str | None
    x_use_raw: bool
    y_use_raw: bool
    x_transform: Callable[[MatrixLike], MatrixLike] | None
    y_transform: Callable[[MatrixLike], MatrixLike] | None


class SpatialKwargs(TypedDict, total=False):
    """The `spatial_pair_proximity` parameters `liana_pipe` forwards."""

    bandwidth: float
    contact_bandwidth: float | None
    min_cells_in_proximity: int
    trim_fraction: float
    kernel: Literal["gaussian", "exponential", "linear", "misty_rbf"]


@d.dedent
def liana_pipe(
    adata: AnnData | MuData,
    groupby: str,
    resource_name: str,
    resource: pd.DataFrame | None,
    interactions: list[tuple[str, str]] | None,
    groupby_pairs: pd.DataFrame | None,
    expr_prop: float,
    min_cells: int,
    base: float,
    de_method: DeMethod,
    n_perms: int | None,
    seed: int,
    verbose: bool,
    use_raw: bool,
    n_jobs: int,
    layer: str | None,
    supp_columns: list[str] | None = None,
    return_all_lrs: bool = False,
    spatial_key: str = "spatial",
    spatial_kwargs: SpatialKwargs | None = None,
    _score: MethodMeta | None = None,
    _methods: Sequence[MethodMeta] | None = None,
    _consensus_opts: list[str] | Literal[False] | None = None,
    _aggregate_method: Literal["rra", "mean"] = "rra",
    mdata_kwargs: MdataKwargs | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Single-cell Ligand-receptor inference pipeline.

    Parameters
    ----------
    %(adata)s
    %(groupby)s
    %(resource_name)s
    %(resource)s
    %(interactions)s
    %(groupby_pairs)s
    %(expr_prop)s
    %(min_cells)s
    %(base)s
    %(de_method)s
    %(n_perms_sc)s
    %(seed)s
    %(verbose)s
    %(use_raw)s
    %(layer)s
    supp_columns
        Additional columns to be added to the output of each method.
    %(return_all_lrs)s
    _score
        Instance of Method classes (None by default - returns LR stats - no methods used).
    _methods
        Methods to be run (only relevant for consensus).
    _consensus_opts
        Ways to aggregate interactions across methods by default does all aggregations (['Specificity', 'Magnitude']).
    _aggregate_method
        RobustRankAggregate('rra') or mean rank ('mean').
    %(spatial_key)s
    %(spatial_kwargs)s
    %(mdata_kwargs)s

    Returns
    -------
    A DataFrame with ligand-receptor results

    """
    if mdata_kwargs is None:
        mdata_kwargs = MdataKwargs()

    _key_cols = P.primary

    if _score is not None:
        _complex_cols, _add_cols = _score.complex_cols, _score.add_cols
    else:
        _complex_cols = [C.ligand_means, C.receptor_means]
        _add_cols = M.get_all_values()

    if n_perms is None:
        _consensus_opts = ["Magnitude"]

    if supp_columns is None:
        supp_columns = []
    _add_cols = _add_cols + [P.ligand, P.receptor, C.ligand_props, C.receptor_props] + supp_columns

    # initialize mat_mean for sca
    mat_mean = None
    mat_max = None

    resource = _handle_resource(
        interactions=interactions, resource=resource, resource_name=resource_name, verbose=verbose
    )
    resource = _explode_complexes(resource)

    if isinstance(adata, MuData):
        adata = mdata_to_anndata(adata, **mdata_kwargs, verbose=verbose)
        use_raw = False
        layer = None
    elif not isinstance(adata, AnnData):
        raise TypeError(f"`adata` must be an AnnData or MuData, got {type(adata).__name__}.")

    groupby_subset = _get_groupby_subset(groupby_pairs=groupby_pairs)
    adata = prep_check_adata(
        adata=adata,
        groupby=groupby,
        groupby_subset=groupby_subset,
        min_cells=min_cells,
        use_raw=use_raw,
        layer=layer,
        obsm=adata.obsm if spatial_key else None,
        verbose=verbose,
    )

    if M.mat_mean in _add_cols:
        mat_mean = np.float32(get_x(adata).mean(dtype="float32"))

    # get mat max for CellChat
    if M.mat_max in _add_cols:
        mat_max = np.float32(get_x(adata).max())

    # Check overlap between resource and adata
    assert_covered(
        np.union1d(np.unique(resource[P.ligand]), np.unique(resource[P.receptor])), adata.var_names, verbose=verbose
    )

    # Filter Resource
    resource = filter_resource(resource, adata.var_names)

    # Cluster stats
    if (M.ligand_cdf in _add_cols) or (M.receptor_cdf in _add_cols):
        cluster_stats = _cluster_stats(adata)

    # Create Entities
    entities = np.union1d(np.unique(resource[P.ligand]), np.unique(resource[P.receptor]))
    # Filter to only include the relevant genes
    adata = adata[:, np.intersect1d(entities, adata.var.index)]

    if verbose:
        print(f"Generating ligand-receptor stats for {adata.shape[0]} samples and {adata.shape[1]} features")

    # Get lr results
    lr_res = _get_lr(
        adata=adata,
        resource=resource,
        groupby_pairs=groupby_pairs,
        mat_mean=mat_mean,
        mat_max=mat_max,
        relevant_cols=_key_cols + _add_cols + _complex_cols,
        de_method=de_method,
        base=base,
        verbose=verbose,
    )

    # Ligand and receptor score based on unfiltered cluster mean and cluster std. Handles protein complexes
    if (M.ligand_cdf in _add_cols) or (M.receptor_cdf in _add_cols):
        lr_res = _complex_score(lr_res, cluster_stats)

    # Mean Sums required for NATMI (note done on subunits also)
    if M.ligand_means_sums in _add_cols:
        on = [x for x in P.complete if x != P.source]
        lr_res = _sum_means(lr_res, what=C.ligand_means, on=on)
    if M.receptor_means_sums in _add_cols:
        on = [x for x in P.complete if x != P.target]
        lr_res = _sum_means(lr_res, what=C.receptor_means, on=on)

    # Calculate spatial proximity if available and add to lr_res
    if spatial_key in adata.obsm:
        if spatial_kwargs is None:
            spatial_kwargs = SpatialKwargs()

        proximity_df = spatial_pair_proximity(
            adata=adata, groupby="@label", spatial_key=spatial_key, verbose=verbose, **spatial_kwargs
        )
        # Set interacting to 0 where not interacting
        # Note: this sets proximity to 0 for non-interacting pairs (according to the count/interaction
        # threshold used inside spatial_pair_proximity), so the resulting "proximity" column reflects
        # both spatial closeness AND interaction/count-based significance rather than pure distance.
        proximity_df["proximity"] = proximity_df["proximity"] * proximity_df["interacting"]

        lr_res = lr_res.merge(proximity_df[["source", "target", "proximity"]], on=["source", "target"], how="left")
        lr_res["proximity"] = lr_res["proximity"].fillna(0.0)

    # Calculate Score
    if _score is not None:
        if _score.method_name == "Rank_Aggregate":
            # Run all methods in consensus
            lrs: dict[str, pd.DataFrame] = {}
            if _methods is None:
                raise ValueError("`_methods` must be provided for the consensus method.")
            for method in _methods:
                if verbose:
                    print(f"Running {method.method_name}")

                lrs[method.method_name] = _run_method(
                    lr_res=lr_res.copy(),
                    adata=adata,
                    groupby=groupby,
                    expr_prop=expr_prop,
                    _score=method,
                    _key_cols=_key_cols,
                    _complex_cols=method.complex_cols,
                    _add_cols=method.add_cols,
                    n_perms=n_perms,
                    seed=seed,
                    return_all_lrs=return_all_lrs,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    _aggregate_flag=True,
                )
            if _consensus_opts is False:  # Return by method results as they are
                return lrs
            # local import: `_rank_aggregate` imports this module, so a module-level
            # import here would close a circular import
            from liana.method.sc._rank_aggregate import AggregateClass

            if not isinstance(_score, AggregateClass):
                raise TypeError("Aggregation requires an `AggregateClass` score.")
            lr_res = _aggregate(
                lrs,
                consensus=_score,
                aggregate_method=_aggregate_method,
                _key_cols=_key_cols,
                _consensus_opts=_consensus_opts,
            )
        else:  # Run the specific method in mind
            lr_res = _run_method(
                lr_res=lr_res,
                adata=adata,
                groupby=groupby,
                expr_prop=expr_prop,
                _score=_score,
                _key_cols=_key_cols,
                _complex_cols=_complex_cols,
                _add_cols=_add_cols,
                n_perms=n_perms,
                return_all_lrs=return_all_lrs,
                n_jobs=n_jobs,
                verbose=verbose,
                seed=seed,
            )
    else:  # Just return lr_res
        lr_res = _filter_reassemble_complexes(
            lr_res=lr_res,
            _key_cols=_key_cols,
            expr_prop=expr_prop,
            complex_cols=_complex_cols,
            return_all_lrs=return_all_lrs,
        )

    if _score is not None:
        orderby, ascending = (
            (_score.magnitude, _score.magnitude_ascending)
            if _score.magnitude is not None
            else (_score.specificity, _score.specificity_ascending)
        )
        if orderby is None:
            raise ValueError(f"`{_score.method_name}` reports neither a magnitude nor a specificity score.")

        lr_res = lr_res.sort_values(by=orderby, ascending=bool(ascending))

    return lr_res


def _get_lr(
    adata: AnnData,
    resource: pd.DataFrame,
    groupby_pairs: pd.DataFrame | None,
    relevant_cols: list[str],
    mat_mean: np.float32 | None,
    mat_max: np.float32 | None,
    de_method: DeMethod,
    base: float,
    verbose: bool,
) -> pd.DataFrame:
    labels = get_obs(adata)[I.label].cat.categories

    # Method-specific stats
    connectome_flag = (M.ligand_zscores in relevant_cols) | (M.receptor_zscores in relevant_cols)
    if connectome_flag:
        adata.layers["scaled"] = sc.pp.scale(adata, copy=True).X

    logfc_flag = (M.ligand_logfc in relevant_cols) | (M.receptor_logfc in relevant_cols)
    if logfc_flag:
        if "log1p" in adata.uns_keys():
            if (adata.uns["log1p"]["base"] is not None) & verbose:
                print("Assuming that counts were `natural` log-normalized!")
        elif ("log1p" not in adata.uns_keys()) & verbose:
            print("Assuming that counts were `natural` log-normalized!")
        # `prep_check_adata` upstream guarantees a csr matrix.
        normcounts = _choose_mtx_rep(adata).copy()
        normcounts.data = _expm1_base(normcounts.data, base)
        adata.layers["normcounts"] = normcounts

    # initialize dict
    dedict: dict[str, pd.DataFrame] = {}

    # Calc pvals + other stats per gene or not
    rank_genes_bool = (C.ligand_pvals in relevant_cols) | (C.receptor_pvals in relevant_cols)
    if rank_genes_bool:
        ranked = sc.tl.rank_genes_groups(adata, groupby=I.label, method=de_method, use_raw=False, copy=True)
        if ranked is None:
            raise RuntimeError("`scanpy.tl.rank_genes_groups(copy=True)` returned nothing.")
        adata = ranked

    obs = get_obs(adata)
    for label in labels:
        temp = adata[obs[I.label] == label, :]
        a = _get_props(_choose_mtx_rep(temp))
        stats = pd.DataFrame({"names": temp.var_names, "props": a}).assign(label=label).sort_values("names")
        if rank_genes_bool:
            pvals = sc.get.rank_genes_groups_df(adata, label)
            stats = stats.merge(pvals)
        dedict[label] = stats

    # check if genes are ordered correctly
    if not list(adata.var_names) == list(dedict[labels[0]]["names"]):
        raise AssertionError("Variable names did not match DE results!")

    # Calculate Mean, logFC and z-scores by group
    for label in labels:
        temp = adata[get_obs(adata)[I.label].isin([label])]
        dedict[label]["means"] = np.asarray(_choose_mtx_rep(temp).mean(axis=0)).ravel()
        if connectome_flag:
            dedict[label]["zscores"] = temp.layers["scaled"].mean(axis=0)
        if logfc_flag:
            dedict[label]["logfc"] = _calc_log2fc(adata, label)
        if isinstance(mat_max, np.float32):  # cellchat flag
            dedict[label]["trimean"] = _trimean(_choose_mtx_rep(temp) / mat_max)

    pairs = pd.DataFrame(np.array(np.meshgrid(labels, labels)).reshape(2, np.size(labels) * np.size(labels)).T).rename(
        columns={0: P.source, 1: P.target}
    )

    if groupby_pairs is not None:
        pairs = pairs.merge(groupby_pairs, on=[P.source, P.target], how="inner")

    # Join Stats
    lr_res = pd.concat(
        [
            _join_stats(source, target, dedict, resource)
            for source, target in zip(pairs[P.source], pairs[P.target], strict=False)
        ]
    )

    if M.mat_mean in relevant_cols:
        assert isinstance(mat_mean, np.float32)
        lr_res[M.mat_mean] = mat_mean

    # NOTE: this is not needed
    if isinstance(mat_max, np.float32):
        lr_res[M.mat_max] = mat_max

    # subset to only relevant columns
    return lr_res[np.intersect1d(relevant_cols, lr_res.columns)]


def _sum_means(lr_res: pd.DataFrame, what: str, on: list[str]) -> pd.DataFrame:
    return lr_res.join(lr_res.groupby(on)[what].sum(), on=on, rsuffix="_sums")


def _calc_log2fc(adata: AnnData, label: str) -> np.ndarray:
    # Get subject vs rest cells
    subject = adata[adata.obs[I.label].isin([label])]
    rest = adata[~adata.obs[I.label].isin([label])]

    if rest.n_obs == 0:
        raise ValueError(
            f"Cannot compute log2FC for group '{label}': every cell belongs to it, "
            "leaving no cells to compare against. This typically happens when "
            "`sample_key` splits the data such that a sample contains only a single "
            "`groupby` category. Ensure each `sample_key` group contains more than "
            "one `groupby` category."
        )

    # subject and rest means
    subj_means = np.asarray(_choose_mtx_rep(subject, layer="normcounts").mean(axis=0)).ravel()
    rest_means = np.asarray(_choose_mtx_rep(rest, layer="normcounts").mean(axis=0)).ravel()

    # log2 + 1 transform
    subj_log2means = np.log2(subj_means + 1)
    loso_log2means = np.log2(rest_means + 1)

    return np.asarray(subj_log2means - loso_log2means)


def _expm1_base(X: np.ndarray, base: float) -> np.ndarray:
    return np.asarray(np.power(base, X) - 1)


def _run_method(
    lr_res: pandas.DataFrame,
    adata: AnnData,
    groupby: str,
    expr_prop: float,
    _score: MethodMeta,
    _key_cols: list[str],
    _complex_cols: list[str],
    _add_cols: list[str],
    n_perms: int | None,
    seed: int,
    return_all_lrs: bool,
    n_jobs: int,
    verbose: bool,
    _aggregate_flag: bool = False,  # relevant for rank_aggregate
) -> pd.DataFrame:
    # re-assemble complexes - specific for each method
    lr_res = _filter_reassemble_complexes(
        lr_res=lr_res,
        _key_cols=_key_cols,
        expr_prop=expr_prop,
        return_all_lrs=return_all_lrs,
        complex_cols=_complex_cols,
    )

    _add_cols = _add_cols + [P.ligand, P.receptor]
    relevant_cols = reduce(lambda left, right: list(np.union1d(left, right)), [_key_cols, _complex_cols, _add_cols])

    # Preserve proximity column if present
    if "proximity" in lr_res.columns:
        relevant_cols = list(relevant_cols) + ["proximity"]

    if return_all_lrs:
        relevant_cols = list(relevant_cols) + [I.lrs_to_keep]
        # separate those that pass from rest
        rest_res = lr_res[~lr_res[I.lrs_to_keep]]
        rest_res = rest_res[relevant_cols]
        lr_res = lr_res[lr_res[I.lrs_to_keep]]
    lr_res = lr_res[relevant_cols]

    # Extract proximity weights if present in lr_res
    proximity_weights = np.asarray(lr_res["proximity"].to_numpy()) if "proximity" in lr_res.columns else None

    if (M.mat_max in _add_cols) & (_score.method_name == "CellChat"):
        # CellChat matrix_max
        norm_factor = np.unique(lr_res[M.mat_max].to_numpy())[0]
        agg_fn: _AggFn = _trimean  # Calculate sparse matrix quantiles?
    else:
        norm_factor = None
        agg_fn = np.mean  # NOTE: change to sparse matrix mean?

    if _score.fun is None:
        raise ValueError(f"`{_score.method_name}` has no scoring function.")

    if _score.permute:
        # get permutations
        if n_perms is not None:
            perms = _get_means_perms(
                adata=adata,
                n_perms=n_perms,
                seed=seed,
                agg_fn=agg_fn,
                norm_factor=norm_factor,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            # get tensor indexes for ligand, receptor, source, target
            ligand_idx, receptor_idx, source_idx, target_idx = _get_mat_idx(adata, lr_res)

            # ligand and receptor perms
            ligand_stat_perms = perms[:, source_idx, ligand_idx]
            receptor_stat_perms = perms[:, target_idx, receptor_idx]
            # stack them together
            perm_stats = np.stack((ligand_stat_perms, receptor_stat_perms), axis=0)
        else:
            perm_stats = None

        scores = _score.fun(x=lr_res, perm_stats=perm_stats)
    else:  # non-perm funs
        scores = _score.fun(x=lr_res)

        # Apply spatial weighting AFTER scoring for non-permutation methods
        if proximity_weights is not None:
            weighted_magnitude = scores[0] * proximity_weights if scores[0] is not None else None
            weighted_specificity = scores[1] * proximity_weights if scores[1] is not None else None
            scores = (weighted_magnitude, weighted_specificity)

    if _score.magnitude is not None:
        lr_res.loc[:, _score.magnitude] = scores[0]
    if _score.specificity is not None:
        lr_res.loc[:, _score.specificity] = scores[1]

    if return_all_lrs:
        # re-append rest of results
        lr_res = pd.concat([lr_res, rest_res])
        if _score.magnitude is not None:
            fill_value = _assign_min_or_max(lr_res[_score.magnitude], _score.magnitude_ascending)
            lr_res.loc[~lr_res[I.lrs_to_keep], _score.magnitude] = fill_value
        if _score.specificity is not None:
            fill_value = _assign_min_or_max(lr_res[_score.specificity], _score.specificity_ascending)
            lr_res.loc[~lr_res[I.lrs_to_keep], _score.specificity] = fill_value

    score_cols = [name for name in (_score.magnitude, _score.specificity) if name is not None]
    if _aggregate_flag:  # if consensus keep only the keys and the method scores
        lr_res = lr_res[_key_cols + score_cols]
    if _score.specificity is not None:  # when n_perms is None
        if lr_res[_score.specificity].isna().all():
            lr_res = lr_res.drop(_score.specificity, axis=1)

    # Remove proximity column if present (internal use only)
    if "proximity" in lr_res.columns:
        lr_res = lr_res.drop("proximity", axis=1)

    return lr_res


def _assign_min_or_max(x: pd.Series, x_ascending: bool | None) -> float:
    return float(np.max(x) if x_ascending else np.min(x))


def _trimean(a: csr_matrix, axis: int = 0) -> np.ndarray:
    """Tukey's trimean, the location estimate CellChat scores with."""
    dense = a.toarray()
    quantiles = np.quantile(dense, q=[0.25, 0.75], axis=axis)
    median = np.median(dense, axis=axis)
    return np.asarray((quantiles[0] + 2 * median + quantiles[1]) / 4)


def _cluster_stats(adata: AnnData) -> pd.DataFrame:
    obs = get_obs(adata)
    cluster_stats = obs.groupby("@label").size().to_frame(name="counts")
    labels = obs["@label"].cat.categories
    for label in labels:
        temp = _choose_mtx_rep(adata[obs["@label"].isin([label])])

        cluster_stats.loc[label, "mean"] = temp.mean()
        cluster_stats.loc[label, "std"] = np.std(temp.toarray())

    return cluster_stats


def _gene_cdf(
    gene_mean: pd.Series,
    cluster_mean: pd.Series,
    cluster_std: pd.Series,
    cluster_counts: pd.Series,
) -> np.ndarray:
    probability = np.asarray(norm.cdf(gene_mean, loc=cluster_mean, scale=cluster_std / np.sqrt(cluster_counts)))
    probability[gene_mean == 0] = 0

    return probability


def _complex_score(lr_res: pd.DataFrame, cluster_stats: pd.DataFrame) -> pd.DataFrame:
    _lr_res = lr_res.merge(cluster_stats.add_prefix("source_"), left_on="source", right_index=True, how="left")
    _lr_res = _lr_res.merge(cluster_stats.add_prefix("target_"), left_on="target", right_index=True, how="left")

    lr_res["ligand_cdf"] = _gene_cdf(
        _lr_res["ligand_means"], _lr_res["source_mean"], _lr_res["source_std"], _lr_res["source_counts"]
    )
    lr_res["receptor_cdf"] = _gene_cdf(
        _lr_res["receptor_means"], _lr_res["target_mean"], _lr_res["target_std"], _lr_res["target_counts"]
    )

    return lr_res
