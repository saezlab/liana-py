from __future__ import annotations

from functools import reduce

import anndata
import numpy as np
import pandas
import pandas as pd
import scanpy as sc
from mudata import MuData
from scipy.stats import norm

from liana._constants import CommonColumns as C
from liana._constants import InternalValues as I
from liana._constants import MethodColumns as M
from liana._constants import PrimaryColumns as P
from liana._docs import d
from liana.method._pipe_utils import assert_covered, filter_resource, prep_check_adata
from liana.method._pipe_utils._aggregate import _aggregate
from liana.method._pipe_utils._common import _get_groupby_subset, _get_props, _join_stats
from liana.method._pipe_utils._get_mean_perms import _get_mat_idx, _get_means_perms
from liana.resource import explode_complexes, filter_reassemble_complexes
from liana.resource.select_resource import _handle_resource
from liana.utils import mdata_to_anndata
from liana.utils.spatial_neighbors import spatial_pair_proximity


@d.dedent
def liana_pipe(adata: anndata.AnnData,
               groupby: str,
               resource_name: str,
               resource: pd.DataFrame | None,
               interactions,
               groupby_pairs: pd.DataFrame | None,
               expr_prop: float,
               min_cells: int,
               base: float,
               de_method: str,
               n_perms: int,
               seed: int,
               verbose: bool,
               use_raw: bool,
               n_jobs: int,
               layer: str | None,
               supp_columns: list | None = None,
               return_all_lrs: bool = False,
               spatial_key: str = 'spatial',
               spatial_kwargs: dict | None = None,
               _score=None,
               _methods: list = None,
               _consensus_opts: list = None,
               _aggregate_method: str | None = None,
               mdata_kwargs: dict | None = None,
               ):
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
        mdata_kwargs = {}

    _key_cols = P.primary

    if _score is not None:
        _complex_cols, _add_cols = _score.complex_cols, _score.add_cols
    else:
        _complex_cols = [C.ligand_means, C.receptor_means]
        _add_cols = M.get_all_values()

    if n_perms is None:
        _consensus_opts = 'Magnitude'

    if supp_columns is None:
        supp_columns = []
    _add_cols = _add_cols + [P.ligand, P.receptor,
                             C.ligand_props, C.receptor_props] + supp_columns

    # initialize mat_mean for sca
    mat_mean = None
    mat_max = None

    resource = _handle_resource(interactions=interactions,
                                resource=resource,
                                resource_name=resource_name,
                                verbose=verbose)
    resource = explode_complexes(resource)

    if isinstance(adata, MuData):
        adata = mdata_to_anndata(adata, **mdata_kwargs, verbose=verbose)
        use_raw = False
        layer = None

    groupby_subset = _get_groupby_subset(groupby_pairs=groupby_pairs)
    adata = prep_check_adata(adata=adata,
                             groupby=groupby,
                             groupby_subset=groupby_subset,
                             min_cells=min_cells,
                             use_raw=use_raw,
                             layer=layer,
                             obsm=adata.obsm if spatial_key else None,
                             verbose=verbose)

    if M.mat_mean in _add_cols:
        mat_mean = np.mean(adata.X, dtype='float32')

    # get mat max for CellChat
    if M.mat_max in _add_cols:
        mat_max = adata.X.max()
        assert isinstance(mat_max, np.float32)

    # Check overlap between resource and adata
    assert_covered(np.union1d(np.unique(resource[P.ligand]),
                              np.unique(resource[P.receptor])),
                   adata.var_names, verbose=verbose)

    # Filter Resource
    resource = filter_resource(resource, adata.var_names)

    # Cluster stats
    if (M.ligand_cdf in _add_cols) or (M.receptor_cdf in _add_cols):
        cluster_stats = _cluster_stats(adata)

    # Create Entities
    entities = np.union1d(np.unique(resource[P.ligand]),
                          np.unique(resource[P.receptor]))
    # Filter to only include the relevant genes
    adata = adata[:, np.intersect1d(entities, adata.var.index)]

    if verbose:
        print(f"Generating ligand-receptor stats for {adata.shape[0]} samples "
              f"and {adata.shape[1]} features")

    # Get lr results
    lr_res = _get_lr(adata=adata,
                     resource=resource,
                     groupby_pairs=groupby_pairs,
                     mat_mean=mat_mean,
                     mat_max=mat_max,
                     relevant_cols=_key_cols + _add_cols + _complex_cols,
                     de_method=de_method,
                     base=base,
                     verbose=verbose
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
            spatial_kwargs = {}

        proximity_df = spatial_pair_proximity(
            adata=adata,
            groupby='@label',
            spatial_key=spatial_key,
            verbose=verbose,
            **spatial_kwargs
        )
        # Set interacting to 0 where not interacting
        # Note: this sets proximity to 0 for non-interacting pairs (according to the count/interaction
        # threshold used inside spatial_pair_proximity), so the resulting "proximity" column reflects
        # both spatial closeness AND interaction/count-based significance rather than pure distance.
        proximity_df['proximity'] = proximity_df['proximity'] * proximity_df['interacting']

        lr_res = lr_res.merge(
            proximity_df[['source', 'target', 'proximity']],
            on=['source', 'target'],
            how='left'
        )
        lr_res['proximity'] = lr_res['proximity'].fillna(0.0)

    # Calculate Score
    if _score is not None:
        if _score.method_name == "Rank_Aggregate":
            # Run all methods in consensus
            lrs = {}
            for method in _methods:
                if verbose:
                    print(f"Running {method.method_name}")

                lrs[method.method_name] = \
                    _run_method(lr_res=lr_res.copy(),
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
                                _aggregate_flag=True
                                )
            if _consensus_opts is not False:
                lr_res = _aggregate(lrs,
                                    consensus=_score,
                                    aggregate_method=_aggregate_method,
                                    _key_cols=_key_cols,
                                    _consensus_opts=_consensus_opts,
                                    )
            else:  # Return by method results as they are
                return lrs
        else:  # Run the specific method in mind
            lr_res = _run_method(lr_res=lr_res,
                                 adata=adata,
                                 groupby=groupby,
                                 expr_prop=expr_prop,
                                 _score=_score, _key_cols=_key_cols,
                                 _complex_cols=_complex_cols,
                                 _add_cols=_add_cols,
                                 n_perms=n_perms,
                                 return_all_lrs=return_all_lrs,
                                 n_jobs=n_jobs,
                                 verbose=verbose,
                                 seed=seed)
    else:  # Just return lr_res
        lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                             _key_cols=_key_cols,
                                             expr_prop=expr_prop,
                                             complex_cols=_complex_cols,
                                             return_all_lrs=return_all_lrs)

    if _score is not None:
        orderby, ascending =  (_score.magnitude, _score.magnitude_ascending) if _score.magnitude is not None \
            else (_score.specificity, _score.specificity_ascending)

        lr_res = lr_res.sort_values(by=orderby, ascending=ascending)

    return lr_res


def _get_lr(adata, resource, groupby_pairs, relevant_cols, mat_mean, mat_max, de_method, base, verbose):
    labels = adata.obs[I.label].cat.categories

    # Method-specific stats
    connectome_flag = (M.ligand_zscores in relevant_cols) | (
                M.receptor_zscores in relevant_cols)
    if connectome_flag:
        adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X

    logfc_flag = (M.ligand_logfc in relevant_cols) | (
            M.receptor_logfc in relevant_cols)
    if logfc_flag:
        if 'log1p' in adata.uns_keys():
            if (adata.uns['log1p']['base'] is not None) & verbose:
                print("Assuming that counts were `natural` log-normalized!")
        elif ('log1p' not in adata.uns_keys()) & verbose:
            print("Assuming that counts were `natural` log-normalized!")
        adata.layers['normcounts'] = adata.X.copy()
        adata.layers['normcounts'].data = _expm1_base(adata.X.data, base)

    # initialize dict
    dedict = {}

    # Calc pvals + other stats per gene or not
    rank_genes_bool = (C.ligand_pvals in relevant_cols) | (C.receptor_pvals in relevant_cols)
    if rank_genes_bool:
        adata = sc.tl.rank_genes_groups(adata, groupby=I.label,
                                        method=de_method, use_raw=False,
                                        copy=True)

    for label in labels:
        temp = adata[adata.obs[I.label] == label, :]
        a = _get_props(temp.X)
        stats = pd.DataFrame({'names': temp.var_names, 'props': a}). \
            assign(label=label).sort_values('names')
        if rank_genes_bool:
            pvals = sc.get.rank_genes_groups_df(adata, label)
            stats = stats.merge(pvals)
        dedict[label] = stats

    # check if genes are ordered correctly
    if not list(adata.var_names) == list(dedict[labels[0]]['names']):
        raise AssertionError("Variable names did not match DE results!")

    # Calculate Mean, logFC and z-scores by group
    for label in labels:
        temp = adata[adata.obs[I.label].isin([label])]
        dedict[label]['means'] = temp.X.mean(axis=0).A.flatten()
        if connectome_flag:
            dedict[label]['zscores'] = temp.layers['scaled'].mean(axis=0)
        if logfc_flag:
            dedict[label]['logfc'] = _calc_log2fc(adata, label)
        if isinstance(mat_max, np.float32):  # cellchat flag
            dedict[label]['trimean'] = _trimean(temp.X / mat_max)

    pairs = (pd.DataFrame(np.array(np.meshgrid(labels, labels))
                          .reshape(2, np.size(labels) * np.size(labels)).T)
             .rename(columns={0: P.source, 1: P.target}))

    if groupby_pairs is not None:
        pairs = pairs.merge(groupby_pairs, on=[P.source, P.target], how='inner')

    # Join Stats
    lr_res = pd.concat(
        [_join_stats(source, target, dedict, resource) for source, target in
         zip(pairs[P.source], pairs[P.target], strict=False)]
    )

    if M.mat_mean in relevant_cols:
        assert isinstance(mat_mean, np.float32)
        lr_res[M.mat_mean] = mat_mean

    # NOTE: this is not needed
    if isinstance(mat_max, np.float32):
        lr_res[M.mat_max] = mat_max

    # subset to only relevant columns
    relevant_cols = np.intersect1d(relevant_cols, lr_res.columns)

    return lr_res[relevant_cols]


def _sum_means(lr_res, what, on):
    return lr_res.join(lr_res.groupby(on)[what].sum(), on=on, rsuffix='_sums')


def _calc_log2fc(adata, label) -> np.ndarray:
    # Get subject vs rest cells
    subject = adata[adata.obs[I.label].isin([label])]
    rest = adata[~adata.obs[I.label].isin([label])]

    # subject and rest means
    subj_means = subject.layers['normcounts'].mean(0).A.flatten()
    rest_means = rest.layers['normcounts'].mean(0).A.flatten()

    # log2 + 1 transform
    subj_log2means = np.log2(subj_means + 1)
    loso_log2means = np.log2(rest_means + 1)

    logfc_vec = subj_log2means - loso_log2means

    return logfc_vec


def _expm1_base(X, base):
    return np.power(base, X) - 1


def _run_method(lr_res: pandas.DataFrame,
                adata: anndata.AnnData,
                groupby: str,
                expr_prop: float,
                _score,
                _key_cols: list,
                _complex_cols: list,
                _add_cols: list,
                n_perms: int,
                seed: int,
                return_all_lrs: bool,
                n_jobs: int,
                verbose: bool,
                _aggregate_flag: bool = False  # relevant for rank_aggregate
                ) -> pd.DataFrame:
    # re-assemble complexes - specific for each method
    lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                         _key_cols=_key_cols,
                                         expr_prop=expr_prop,
                                         return_all_lrs=return_all_lrs,
                                         complex_cols=_complex_cols)

    _add_cols = _add_cols + [P.ligand, P.receptor]
    relevant_cols = reduce(np.union1d, [_key_cols, _complex_cols, _add_cols])

    # Preserve proximity column if present
    if 'proximity' in lr_res.columns:
        relevant_cols = list(relevant_cols) + ['proximity']

    if return_all_lrs:
        relevant_cols = list(relevant_cols) + [I.lrs_to_keep]
        # separate those that pass from rest
        rest_res = lr_res[~lr_res[I.lrs_to_keep]]
        rest_res = rest_res[relevant_cols]
        lr_res = lr_res[lr_res[I.lrs_to_keep]]
    lr_res = lr_res[relevant_cols]

    # Extract proximity weights if present in lr_res
    proximity_weights = lr_res['proximity'].values if 'proximity' in lr_res.columns else None

    if (M.mat_max in _add_cols) & (_score.method_name == "CellChat"):
        # CellChat matrix_max
        norm_factor = np.unique(lr_res[M.mat_max].values)[0]
        agg_fun = _trimean # Calculate sparse matrix quantiles?
    else:
        norm_factor = None
        agg_fun = np.mean # NOTE: change to sparse matrix mean?

    if _score.permute:
        # get permutations
        if n_perms is not None:
            perms = _get_means_perms(adata=adata,
                                     n_perms=n_perms,
                                     seed=seed,
                                     agg_fun=agg_fun,
                                     norm_factor=norm_factor,
                                     n_jobs=n_jobs,
                                     verbose=verbose)
            # get tensor indexes for ligand, receptor, source, target
            ligand_idx, receptor_idx, source_idx, target_idx = _get_mat_idx(adata, lr_res)

            # ligand and receptor perms
            ligand_stat_perms = perms[:, source_idx, ligand_idx]
            receptor_stat_perms = perms[:, target_idx, receptor_idx]
            # stack them together
            perm_stats = np.stack((ligand_stat_perms, receptor_stat_perms), axis=0)
        else:
            perm_stats = None

        scores = _score.fun(x=lr_res,
                            perm_stats=perm_stats)
    else:  # non-perm funs
        scores = _score.fun(x=lr_res)

        # Apply spatial weighting AFTER scoring for non-permutation methods
        if proximity_weights is not None:
            weighted_magnitude = scores[0] * proximity_weights if scores[0] is not None else None
            weighted_specificity = scores[1] * proximity_weights if scores[1] is not None else None
            scores = (weighted_magnitude, weighted_specificity)

    lr_res.loc[:, _score.magnitude] = scores[0]
    lr_res.loc[:, _score.specificity] = scores[1]

    if return_all_lrs:
        # re-append rest of results
        lr_res = pd.concat([lr_res, rest_res], copy=False)
        if _score.magnitude is not None:
            fill_value = _assign_min_or_max(lr_res[_score.magnitude],
                                            _score.magnitude_ascending)
            lr_res.loc[~lr_res[I.lrs_to_keep], _score.magnitude] = fill_value
        if _score.specificity is not None:
            fill_value = _assign_min_or_max(lr_res[_score.specificity],
                                            _score.specificity_ascending)
            lr_res.loc[~lr_res[I.lrs_to_keep], _score.specificity] = fill_value

    if _aggregate_flag:  # if consensus keep only the keys and the method scores
        lr_res = lr_res[_key_cols + [_score.magnitude, _score.specificity]]

    # remove redundant cols for some scores
    if (_score.magnitude is None) | (_score.specificity is None):
        lr_res = lr_res.drop([None], axis=1)
    if _score.specificity is not None: # when n_perms is None
        if lr_res[_score.specificity].isna().all():
            lr_res = lr_res.drop(_score.specificity, axis=1)

    # Remove proximity column if present (internal use only)
    if 'proximity' in lr_res.columns:
        lr_res = lr_res.drop('proximity', axis=1)

    return lr_res


def _assign_min_or_max(x, x_ascending):
    if x_ascending:
        return np.max(x)
    else:
        return np.min(x)


def _trimean(a, axis=0):
    quantiles = np.quantile(a.toarray(), q=[0.25, 0.75], axis=axis)
    median = np.median(a.toarray(), axis=axis)
    return (quantiles[0] + 2 * median + quantiles[1]) / 4

def _cluster_stats(adata):
    cluster_stats = adata.obs.groupby('@label').size().to_frame(name='counts')
    labels = adata.obs['@label'].cat.categories
    for label in labels:
        temp = adata[adata.obs['@label'].isin([label])]

        cluster_stats.loc[label, 'mean'] = temp.X.mean()
        cluster_stats.loc[label, 'std'] = np.std(temp.X.toarray())

    return cluster_stats


def _gene_cdf(gene_mean, cluster_mean, cluster_std, cluster_counts):
    probability = norm.cdf(gene_mean, loc=cluster_mean, scale = cluster_std / np.sqrt(cluster_counts))
    probability[gene_mean==0] = 0

    return probability

def _complex_score(lr_res, cluster_stats):
    _lr_res = lr_res.merge(cluster_stats.add_prefix('source_'), left_on='source', right_index=True, how='left')
    _lr_res = _lr_res.merge(cluster_stats.add_prefix('target_'), left_on='target', right_index=True, how='left')

    lr_res['ligand_cdf'] = _gene_cdf(_lr_res['ligand_means'],
                                       _lr_res['source_mean'],
                                       _lr_res['source_std'],
                                       _lr_res['source_counts'])
    lr_res['receptor_cdf'] = _gene_cdf(_lr_res['receptor_means'],
                                         _lr_res['target_mean'],
                                         _lr_res['target_std'],
                                         _lr_res['target_counts'])

    return lr_res
