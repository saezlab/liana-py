from __future__ import annotations

import anndata
import pandas

from liana.method._pipe_utils import prep_check_adata, assert_covered, filter_resource, \
    filter_reassemble_complexes
from ..resource import select_resource, explode_complexes
from liana.method._pipe_utils._get_mean_perms import _get_means_perms
from liana.method._pipe_utils._aggregate import _aggregate

import scanpy as sc
import pandas as pd
import numpy as np
from functools import reduce


def liana_pipe(adata: anndata.AnnData,
               groupby: str,
               resource_name: str,
               resource: pd.DataFrame | None,
               expr_prop: float,
               min_cells: int,
               base: float,
               de_method: str,
               n_perms: int,
               seed: int,
               verbose: bool,
               use_raw: bool,
               layer: str | None,
               supp_cols: list | None = None,
               _key_cols: list = None,
               _score=None,
               _methods: list = None,
               _consensus_opts: list = None,
               _aggregate_method: str = None,
               _return_subunits: bool = False
               ):
    """
    Parameters
    ----------
    adata
        Annotated data object.
    groupby
        The key of the observations grouping to consider.
    resource_name
        Name of the resource to be loaded and use for ligand-receptor inference.
    resource
        Parameter to enable external resources to be passed. Expects a pandas dataframe
        with [`ligand`, `receptor`] columns. None by default. If provided will overrule
        the resource requested via `resource_name`
    expr_prop
        Minimum expression proportion for the ligands/receptors (and their subunits) in the
         corresponding cell identities. Set to `0`, to return unfiltered results.
    min_cells
        Minimum cells per cell identity
    base
        The base by which to do expm1 (relevant only for 1vsRest logFC calculation)
    de_method
        Differential expression method. `scanpy.tl.rank_genes_groups` is used to rank genes
        according to 1vsRest. The default method is 't-test'. Only relevant if p-values
        are included in `supp_cols`
    n_perms
        n permutations (relevant only for permutation-based methods)
    seed
        Random seed for reproducibility 
    verbose
        Verbosity flag
    use_raw
        Use raw attribute of adata if present.
    layer
        Layer in anndata.AnnData.layers to use. If None, use anndata.AnnData.X.
    supp_cols
        additional columns to be added to the output of each method.
    _key_cols
        columns which make every interaction unique (i.e. PK).
    _score
        Instance of Method classes (None by default - returns LR stats - no methods used).
    _methods
        Methods to be run (only relevant for consensus).
    _consensus_opts
        Ways to aggregate interactions across methods by default does all aggregations (['Steady',
        'Specificity', 'Magnitude']).
    _aggregate_method
        RobustRankAggregate('rra') or mean rank ('mean').
    _return_subunits
        Whether to return only the subunits (False by default).

    Returns
    -------
    A adata frame with ligand-receptor results

    """
    if _key_cols is None:
        _key_cols = ['source', 'target', 'ligand_complex', 'receptor_complex']

    if _score is not None:
        _complex_cols, _add_cols = _score.complex_cols, _score.add_cols
    else:
        _complex_cols = ['ligand_means', 'receptor_means']
        # change to full list and move to _var
        _add_cols = ['ligand_means_sums', 'receptor_means_sums',
                     'ligand_zscores', 'receptor_zscores',
                     'ligand_logfc', 'receptor_logfc',
                     'mat_mean',
                     ]

    if supp_cols is None:
        supp_cols = []
    _add_cols = _add_cols + ['ligand', 'receptor', 'ligand_props', 'receptor_props'] + supp_cols

    # Check and Reformat Mat if needed
    adata = prep_check_adata(adata=adata,
                             groupby=groupby,
                             min_cells=min_cells,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose)

    # get mat mean for SCA
    if 'mat_mean' in _add_cols:
        adata.uns['mat_mean'] = np.mean(adata.X)

    if resource is None:
        resource = select_resource(resource_name.lower())
    # explode complexes/decomplexify
    resource = explode_complexes(resource)
    # Filter Resource
    resource = filter_resource(resource, adata.var_names)

    # Create Entities
    entities = np.union1d(np.unique(resource["ligand"]),
                          np.unique(resource["receptor"]))

    # Check overlap between resource and adata
    assert_covered(entities, adata.var_names, verbose=verbose)

    # Filter to only include the relevant genes
    adata = adata[:, np.intersect1d(entities, adata.var.index)]

    # Get lr results
    lr_res = _get_lr(adata, resource,
                     _key_cols + _add_cols + _complex_cols,
                     de_method, base, verbose)

    # Mean Sums required for NATMI (note done on subunits also)
    if 'ligand_means_sums' in _add_cols:
        lr_res = _sum_means(lr_res, what='ligand_means',
                            on=['ligand_complex', 'receptor_complex',
                                'ligand', 'receptor', 'target'])
    if 'receptor_means_sums' in _add_cols:
        lr_res = _sum_means(lr_res, what='receptor_means',
                            on=['ligand_complex', 'receptor_complex',
                                'ligand', 'receptor', 'source'])

    # Calculate Score
    if _score is not None:
        if _score.method_name == "Rank_Aggregate":
            # Run all methods in consensus
            lrs = {}
            for method in _methods:
                if verbose:
                    f"Running {method.method_name}."

                lrs[method.method_name] = \
                    _run_method(lr_res=lr_res.copy(),
                                adata=adata,
                                expr_prop=expr_prop,
                                _score=method,
                                _key_cols=_key_cols,
                                _complex_cols=method.complex_cols,
                                _add_cols=method.add_cols,
                                n_perms=n_perms,
                                seed=seed,
                                _consensus=True
                                )
            if _consensus_opts is not False:
                lr_res = _aggregate(lrs,
                                    consensus=_score,
                                    aggregate_method=_aggregate_method,
                                    _key_cols=_key_cols)
            else:  # Return by method results as they are
                return lrs
        else:  # Run the specific method in mind
            lr_res = _run_method(lr_res=lr_res, adata=adata, expr_prop=expr_prop,
                                 _score=_score, _key_cols=_key_cols, _complex_cols=_complex_cols,
                                 _add_cols=_add_cols, n_perms=n_perms, seed=seed)
    else:  # Just return lr_res
        if _return_subunits:
            return lr_res
        # else re-asemble subunits into complexes
        lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                             _key_cols=_key_cols,
                                             expr_prop=expr_prop,
                                             complex_cols=_complex_cols)

    return lr_res


def _join_stats(source, target, dedict, resource):
    """
    Joins and renames source-ligand and target-receptor stats to the ligand-receptor resource

    Parameters
    ----------
    source
        Source/Sender cell type
    target
        Target/Receiver cell type
    dedict
        dictionary
    resource
        Ligand-receptor Resource

    Returns
    -------
    Ligand-Receptor stats

    """
    source_stats = dedict[source].copy()
    source_stats.columns = source_stats.columns.map(
        lambda x: 'ligand_' + str(x))
    source_stats = source_stats.rename(
        columns={'ligand_names': 'ligand', 'ligand_label': 'source'})

    target_stats = dedict[target].copy()
    target_stats.columns = target_stats.columns.map(
        lambda x: 'receptor_' + str(x))
    target_stats = target_stats.rename(
        columns={'receptor_names': 'receptor', 'receptor_label': 'target'})

    bound = resource.merge(source_stats).merge(target_stats)

    return bound


def _get_lr(adata, resource, relevant_cols, de_method, base, verbose):
    """
    Run DE analysis and merge needed information with resource for LR inference

    Parameters
    ----------
    adata : anndata.AnnData
        adata filtered and formated to contain only the relevant features for lr inference

    relevant_cols : list
        Relevant column names

    resource : pandas.core.frame.DataFrame formatted and filtered resource
    dataframe with the following columns: [interaction, ligand, receptor,
    ligand_complex, receptor_complex]

    de_method : Literal name of the DE `method` to call via
    scanpy.tl.rank_genes.groups - available options are: ['logreg', 't-test',
    'wilcoxon', 't-test_overestim_var']

    Returns
    -------
    lr_res : pandas.core.frame.DataFrame long-format pandas dataframe with stats
    for all interactions /w matching variables in the dataset.

    """
    # get label cats
    adata = adata.copy()
    labels = adata.obs.label.cat.categories

    # Method-specific stats
    connectome_flag = ('ligand_zscores' in relevant_cols) | (
            'receptor_zscores' in relevant_cols)
    if connectome_flag:
        adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X

    logfc_flag = ('ligand_logfc' in relevant_cols) | (
            'receptor_logfc' in relevant_cols)
    if logfc_flag:
        if 'log1p' in adata.uns_keys():
            if (adata.uns['log1p']['base'] is not None) & verbose:
                print("Assuming that counts were natural log-normalized!")
        elif ('log1p' not in adata.uns_keys()) & verbose:
            print("Assuming that counts were natural log-normalized!")
        adata.layers['normcounts'] = adata.X.copy()
        adata.layers['normcounts'].data = _expm1_base(adata.X.data, base)

    # Get Props
    dedict = {}

    # Calc pvals + other stats per gene or not
    rank_genes_bool = ('ligand_pvals' in relevant_cols) | ('receptor_pvals' in relevant_cols)
    if rank_genes_bool:
        sc.tl.rank_genes_groups(adata, groupby='label',
                                method=de_method, use_raw=False)

    for label in labels:
        temp = adata[adata.obs.label == label, :]
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
        temp = adata[adata.obs.label.isin([label])].copy()
        dedict[label]['means'] = temp.X.mean(0).A.flatten()
        if connectome_flag:
            dedict[label]['zscores'] = temp.layers['scaled'].mean(0)
        if logfc_flag:
            dedict[label]['logfc'] = _calc_log2fc(adata, label)

    # Create df /w cell identity pairs
    pairs = (pd.DataFrame(np.array(np.meshgrid(labels, labels))
                          .reshape(2, np.size(labels) * np.size(labels)).T)
             .rename(columns={0: "source", 1: "target"}))

    # Join Stats
    lr_res = pd.concat(
        [_join_stats(source, target, dedict, resource) for source, target in
         zip(pairs['source'], pairs['target'])]
    )

    if 'mat_mean' in relevant_cols:
        lr_res['mat_mean'] = adata.uns['mat_mean']

    # subset to only relevant columns and return (SIMPLY?)
    relevant_cols = np.intersect1d(relevant_cols, lr_res.columns)

    return lr_res[relevant_cols]


def _sum_means(lr_res, what, on):
    """
    Calculate Sum Means

    Parameters
    ---------
    lr_res
        lr_res with reassembled complexes
    what
        [entity]_means_sums for which the sum is calculated
    on
        columns by which to group and sum

    Returns
    -------
    lr_res with [entity]_means_sums column
    """
    return lr_res.join(lr_res.groupby(on)[what].sum(), on=on, rsuffix='_sums')


def _calc_log2fc(adata, label) -> np.ndarray:
    """
    Calculate 1 vs rest Log2FC for a particular cell identity

    Parameters
    ---------

    adata
        anndata with feature-space reduced to the vars intersecting with the IDs in the resource
    label
        cell identity

    Returns
    -------
    An array with logFC changes

    """
    # Get subject vs rest cells
    subject = adata[adata.obs.label.isin([label])].copy()
    rest = adata[~adata.obs.label.isin([label])].copy()

    # subject and rest means
    subj_means = subject.layers['normcounts'].mean(0).A.flatten()
    rest_means = rest.layers['normcounts'].mean(0).A.flatten()

    # log2 + 1 transform
    subj_log2means = np.log2(subj_means + 1)
    loso_log2means = np.log2(rest_means + 1)

    logfc_vec = subj_log2means - loso_log2means

    return logfc_vec


# Exponent with a custom base
def _expm1_base(X, base):
    return np.power(base, X) - 1


def _run_method(lr_res: pandas.DataFrame,
                adata: anndata.AnnData,
                expr_prop: float,
                _score,
                _key_cols: list,
                _complex_cols: list,
                _add_cols: list,
                n_perms: int,
                seed: int,
                _consensus: bool = False  # Indicates whether we're generating the consensus
                ) -> pd.DataFrame:
    # re-assemble complexes - specific for each method
    lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                         _key_cols=_key_cols,
                                         expr_prop=expr_prop,
                                         complex_cols=_complex_cols)

    # subset to only relevant columns and return (SIMPLY?)
    _add_cols = _add_cols + ['ligand', 'receptor']
    relevant_cols = reduce(np.union1d, [_key_cols, _complex_cols, _add_cols])
    lr_res = lr_res[relevant_cols]

    if _score.permute:
        perms, ligand_pos, receptor_pos, labels_pos = \
            _get_means_perms(adata=adata, lr_res=lr_res, n_perms=n_perms, seed=seed)
        # SHOULD VECTORIZE THE APPLY / w NUMBA !!!
        lr_res[[_score.magnitude, _score.specificity]] = \
            lr_res.apply(_score.fun, axis=1, result_type="expand",
                         perms=perms, ligand_pos=ligand_pos,
                         receptor_pos=receptor_pos, labels_pos=labels_pos)
    else:  # non-perm funs
        lr_res[[_score.magnitude, _score.specificity]] = \
            lr_res.apply(_score.fun, axis=1, result_type="expand")

    if _consensus:  # if consensus keep only the keys and the method scores
        lr_res = lr_res[_key_cols + [_score.magnitude, _score.specificity]]

    # remove redundant cols for some scores
    if (_score.magnitude is None) | (_score.specificity is None):
        lr_res = lr_res.drop([None], axis=1)

    return lr_res


# Function to get gene expr proportions
def _get_props(X_mask):
    return X_mask.getnnz(axis=0) / X_mask.shape[0]
