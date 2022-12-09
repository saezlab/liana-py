from __future__ import annotations

import anndata
import pandas

from liana.method._pipe_utils import prep_check_adata, assert_covered, filter_resource, \
    filter_reassemble_complexes
from liana.utils._utils import _get_props
from ..resource import select_resource
from ._pipe_utils._reassemble_complexes import explode_complexes
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
               supp_columns: list | None = None,
               return_all_lrs: bool = False,
               _key_cols: list = None,
               _score=None,
               _methods: list = None,
               _consensus_opts: list = None,
               _aggregate_method: str = None
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
    supp_columns
        additional columns to be added to the output of each method.
    return_all_lrs
        Bool whether to return all LRs, or only those that surpass the expr_prop threshold.
        `False` by default.
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
                     'ligand_trimean', 'receptor_trimean',
                     'mat_mean', 'mat_max',
                     ]

    if supp_columns is None:
        supp_columns = []
    _add_cols = _add_cols + ['ligand', 'receptor', 'ligand_props', 'receptor_props'] + supp_columns

    # initialize mat_mean for sca
    mat_mean = None
    mat_max = None

    # Check and Reformat Mat if needed
    assert groupby is not None
    adata = prep_check_adata(adata=adata,
                             groupby=groupby,
                             min_cells=min_cells,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose)

    # get mat mean for SCA (before reducing the features)
    if 'mat_mean' in _add_cols:
        mat_mean = np.mean(adata.X, dtype='float32')

    # get mat max for CellChat
    if 'mat_max' in _add_cols:
        mat_max = np.max(adata.X.data)
        assert isinstance(mat_max, np.float32)

    if resource is None:
        resource = select_resource(resource_name.lower())
    # explode complexes/decomplexify
    resource = explode_complexes(resource)

    # Check overlap between resource and adata
    assert_covered(np.union1d(np.unique(resource["ligand"]),
                              np.unique(resource["receptor"])),
                   adata.var_names, verbose=verbose)

    # Filter Resource
    resource = filter_resource(resource, adata.var_names)

    print(f"Generating ligand-receptor stats for {adata.shape[0]} samples "
          f"and {adata.shape[1]} features")

    # Create Entities
    entities = np.union1d(np.unique(resource["ligand"]),
                          np.unique(resource["receptor"]))

    # Filter to only include the relevant genes
    adata = adata[:, np.intersect1d(entities, adata.var.index)]

    # Get lr results
    lr_res = _get_lr(adata=adata, resource=resource,
                     mat_mean=mat_mean, mat_max=mat_max,
                     relevant_cols=_key_cols + _add_cols + _complex_cols,
                     de_method=de_method, base=base, verbose=verbose)

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
                    print(f"Running {method.method_name}")

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
                                return_all_lrs=return_all_lrs,
                                verbose=verbose,
                                _aggregate_flag=True
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
                                 _add_cols=_add_cols, n_perms=n_perms,
                                 return_all_lrs=return_all_lrs,
                                 verbose=verbose, seed=seed)
    else:  # Just return lr_res
        lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                             _key_cols=_key_cols,
                                             expr_prop=expr_prop,
                                             complex_cols=_complex_cols,
                                             return_all_lrs=return_all_lrs)

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


def _get_lr(adata, resource, relevant_cols, mat_mean, mat_max, de_method, base, verbose):
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
                print("Assuming that counts were `natural` log-normalized!")
        elif ('log1p' not in adata.uns_keys()) & verbose:
            print("Assuming that counts were `natural` log-normalized!")
        adata.layers['normcounts'] = adata.X.copy()
        adata.layers['normcounts'].data = _expm1_base(adata.X.data, base)

    # initialize dict
    dedict = {}

    # Calc pvals + other stats per gene or not
    rank_genes_bool = ('ligand_pvals' in relevant_cols) | ('receptor_pvals' in relevant_cols)
    if rank_genes_bool:
        adata = sc.tl.rank_genes_groups(adata, groupby='label',
                                        method=de_method, use_raw=False,
                                        copy=True)

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
        temp = adata[adata.obs.label.isin([label])]
        dedict[label]['means'] = temp.X.mean(axis=0).A.flatten()
        if connectome_flag:
            dedict[label]['zscores'] = temp.layers['scaled'].mean(axis=0)
        if logfc_flag:
            dedict[label]['logfc'] = _calc_log2fc(adata, label)
        if isinstance(mat_max, np.float32):  # cellchat flag
            dedict[label]['trimean'] = _trimean(temp.X / mat_max)

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
        assert isinstance(mat_mean, np.float32)
        lr_res['mat_mean'] = mat_mean

    if isinstance(mat_max, np.float32):
        lr_res['mat_max'] = mat_max

    # subset to only relevant columns
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
    subject = adata[adata.obs.label.isin([label])]
    rest = adata[~adata.obs.label.isin([label])]

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
                return_all_lrs: bool,
                verbose: bool,
                _aggregate_flag: bool = False  # Indicates whether we're generating the consensus
                ) -> pd.DataFrame:
    # re-assemble complexes - specific for each method
    lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                         _key_cols=_key_cols,
                                         expr_prop=expr_prop,
                                         return_all_lrs=return_all_lrs,
                                         complex_cols=_complex_cols)

    _add_cols = _add_cols + ['ligand', 'receptor']
    relevant_cols = reduce(np.union1d, [_key_cols, _complex_cols, _add_cols])
    if return_all_lrs:
        relevant_cols = list(relevant_cols) + ['lrs_to_keep']
        # separate those that pass from rest
        rest_res = lr_res[~lr_res['lrs_to_keep']]
        rest_res = rest_res[relevant_cols]
        lr_res = lr_res[lr_res['lrs_to_keep']]
    lr_res = lr_res[relevant_cols]

    if ('mat_max' in _add_cols) & (_score.method_name == "CellChat"):
        # CellChat matrix_max
        norm_factor = np.unique(lr_res['mat_max'].values).item()
        agg_fun = _trimean
    else:
        norm_factor = None
        agg_fun = np.mean

    if _score.permute:
        perms, ligand_pos, receptor_pos, labels_pos = \
            _get_means_perms(adata=adata,
                             lr_res=lr_res,
                             n_perms=n_perms,
                             seed=seed,
                             agg_fun=agg_fun,
                             norm_factor=norm_factor,
                             verbose=verbose)
        lr_res[[_score.magnitude, _score.specificity]] = \
            lr_res.apply(_score.fun, axis=1, result_type="expand",
                         perms=perms, ligand_pos=ligand_pos,
                         receptor_pos=receptor_pos, labels_pos=labels_pos)
    else:  # non-perm funs
        lr_res[[_score.magnitude, _score.specificity]] = \
            lr_res.apply(_score.fun, axis=1, result_type="expand")

    if return_all_lrs:
        # re-append rest of results
        lr_res = pd.concat([lr_res, rest_res])
        if _score.magnitude is not None:

            fill_value = _assign_min_or_max(lr_res[_score.magnitude],
                                            _score.magnitude_ascending)
            lr_res[_score.magnitude][~lr_res['lrs_to_keep']] = fill_value
        if _score.specificity is not None:
            fill_value = _assign_min_or_max(lr_res[_score.specificity],
                                            _score.specificity_ascending)
            lr_res[_score.specificity][~lr_res['lrs_to_keep']] = fill_value

    if _aggregate_flag:  # if consensus keep only the keys and the method scores
        lr_res = lr_res[_key_cols + [_score.magnitude, _score.specificity]]

    # remove redundant cols for some scores
    if (_score.magnitude is None) | (_score.specificity is None):
        lr_res = lr_res.drop([None], axis=1)

    return lr_res


def _assign_min_or_max(x, x_ascending):
    if x_ascending:
        return np.max(x)
    else:
        return np.min(x)


# Function to get gene expr proportions
def _get_props(X_mask):
    return X_mask.getnnz(axis=0) / X_mask.shape[0]


def _trimean(a, axis=0):
    """
    Tukey's mean / Trimean
    Parameters
    ----------
    a
        array (cell ident counts)

    Returns
    -------
    Trimean value

    """
    return np.mean(
        np.quantile(a.A,
                    q=[0.25, 0.5, 0.5, 0.75],
                    axis=axis),
        axis=axis)
