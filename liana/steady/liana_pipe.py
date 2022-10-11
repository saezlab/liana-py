from ..utils import prep_check_adata, check_if_covered, format_vars, filter_resource,\
    reassemble_complexes
from ..resource import select_resource, explode_complexes
from ._permutations import get_means_perms
from functools import reduce
from .aggregate import aggregate

import scanpy as sc
import pandas as pd
import numpy as np


def liana_pipe(adata, groupby, resource_name, resource, de_method,
               n_perms, seed, verbose, use_raw, layer,
               base=2.718281828459045, _key_cols=None, _score=None,
               _methods=None, _consensus_opts=None, _aggregate_method=None):
    """
    :param adata: adata
    :param groupby: label to group_by
    :param resource_name: resource name
    :param resource: a resource dataframe in liana format
    :param de_method: method to do between cluster DE
    :param n_perms: n permutations (relevant only for permutation-based methods)
    :param seed: random seed
    :param verbose: True/False
    :param base: base used for 1vsRest logFC calculation (natural exponent by default)
    :param layer: typing.Union[str, NoneType], optional (default: None)
    Key from `adata.layers` whose value will be used to perform tests on.
    :param use_raw: typing.Union[bool, NoneType], optional (default: None)
    Use `raw` attribute of `adata` if present.
    :param _key_cols: columns which make every interaction unique (i.e. PK)
    :param _score: Instance of Method classes (None by default - returns LR stats - no methods used)
    :param _methods: Methods to be run (only relevant for consensus)
    :param _consensus_opts: Ways to aggregate interactions across methods by
    default does all aggregations (['Steady', 'Specificity', 'Magnitude']_
    :param _aggregate_method: RobustRankAggregate('rra') or mean rank ('mean')
    :return: Returns an anndata with 'liana_res' in .uns
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
    _add_cols = _add_cols + ['ligand', 'receptor']

    # Check and Reformat Mat if needed
    adata = prep_check_adata(adata,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose)
    # Define idents col name
    adata.obs['label'] = adata.obs[groupby]

    # Remove underscores from gene names
    adata.var_names = format_vars(adata.var_names)

    # get mat mean for SCA
    if 'mat_mean' in _add_cols:
        adata.uns['mat_mean'] = np.mean(adata.X)

    if resource is None:
        resource = select_resource(resource_name)
    # explode complexes/decomplexify
    resource = explode_complexes(resource)
    # Filter Resource
    resource = filter_resource(resource, adata.var_names)

    # Create Entities
    entities = np.union1d(np.unique(resource["ligand"]),
                          np.unique(resource["receptor"]))

    # Check overlap between resource and adata
    check_if_covered(entities, adata.var_keys, verbose=verbose)

    # Filter to only include the relevant genes
    adata = adata[:, np.intersect1d(entities, adata.var.index)]

    # Get lr results
    lr_res = _get_lr(adata, resource,
                     _key_cols + _add_cols + _complex_cols,
                     de_method, base, verbose)

    # Mean Sums required for NATMI
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
        if _score.method_name == "Consensus":
            # Run all methods in consensus
            lrs = {}
            for method in _methods:
                print(method.method_name)
                lrs[method.method_name] = \
                    _run_method(lr_res.copy(),
                                adata,
                                _score=method,
                                _key_cols=_key_cols,
                                _complex_cols=method.complex_cols,
                                _add_cols=method.add_cols + ['ligand',
                                                             'receptor'],
                                n_perms=n_perms, seed=seed, _consensus=True
                                )
            if _consensus_opts is not False:
                lr_res = aggregate(lrs,
                                   consensus=_score,
                                   aggregate_method=_aggregate_method,
                                   _key_cols=_key_cols)
            else:
                return lrs
        else:  # Run the specific method in mind
            lr_res = _run_method(lr_res, adata,
                                 _score, _key_cols, _complex_cols, _add_cols,
                                 n_perms, seed)
    else:  # Just return lr_res
        lr_res = reassemble_complexes(lr_res, _key_cols, _complex_cols)

    return lr_res


# Function to join source and target stats
def _join_stats(source, target, dedict, resource):
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
        list of relevant column names

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
    # Calc DE stats (change to a placeholder that is populated, if not required)
    sc.tl.rank_genes_groups(adata, groupby='label', method=de_method, use_raw=False)
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
                print("Assuming that counts were natural log-normalized!")
        elif ('log1p' not in adata.uns_keys()) & verbose:
            print("Assuming that counts were natural log-normalized!")
        adata.layers['normcounts'] = adata.X.copy()
        adata.layers['normcounts'].data = expm1_base(adata.X.data, base)

    # Get DEGs
    dedict = {label: sc.get.rank_genes_groups_df(adata, label).assign(
        label=label).sort_values('names') for label in labels}

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
            dedict[label]['logfc'] = _calc_log2(adata, label)

    # Create df /w cell identity pairs
    pairs = (pd.DataFrame(np.array(np.meshgrid(labels, labels))
                          .reshape(2, np.size(labels) * np.size(labels)).T)
             .rename(columns={0: "source", 1: "target"}))

    # Join Stats
    lr_res = pd.concat(
        [_join_stats(source, target, dedict, resource) for source, target in
         zip(pairs['source'], pairs['target'])]
    )

    if 'mat_mean' in relevant_cols:  # SHOULD BE METHOD NAME?
        lr_res['mat_mean'] = adata.uns['mat_mean']

    # subset to only relevant columns and return (SIMPLY?)
    relevant_cols = np.intersect1d(relevant_cols, lr_res.columns)

    return lr_res[relevant_cols]


# Function to Sum Means
def _sum_means(lr_res, what, on):
    """
    :param lr_res: recomplexified lr_res
    :param what: [entity]_means_sums for which the sum is calculated
    :param on: columns by which to group and sum
    :return: returns lr_res with [entity]_means_sums column
    """
    return lr_res.join(lr_res.groupby(on)[what].sum(), on=on, rsuffix='_sums')


def _calc_log2(adata, label):
    """
    Calculate 1 vs rest log2fc for a particular cell identity

    :param  adata with feature-space reduced to the vars intersecting
    with the IDs in the resource
    :param label: cell identity
    :return: returns a vector of logFC values for each var in adata
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
def expm1_base(X, base):
    return np.power(base, X) - 1


def _run_method(lr_res, adata, _score, _key_cols, _complex_cols, _add_cols,
                n_perms, seed, _consensus=False):
    # re-assemble complexes - specific for each method
    lr_res = reassemble_complexes(lr_res, _key_cols, _complex_cols)

    # subset to only relevant columns and return (SIMPLY?)
    _add_cols = _add_cols + ['ligand', 'receptor']
    relevant_cols = reduce(np.union1d, [_key_cols, _complex_cols, _add_cols])
    lr_res = lr_res[relevant_cols]

    if _score.permute:
        perms, ligand_pos, receptor_pos, labels_pos = \
            get_means_perms(adata=adata, lr_res=lr_res, n_perms=n_perms, seed=seed)
        # SHOULD VECTORIZE THE APPLY / w NUMBA !!!
        lr_res[[_score.magnitude, _score.specificity]] = \
            lr_res.apply(_score.fun, axis=1, result_type="expand",
                         perms=perms, ligand_pos=ligand_pos,
                         receptor_pos=receptor_pos, labels_pos=labels_pos)
    else:  # non-perm funs
        lr_res[[_score.magnitude, _score.specificity]] = \
            lr_res.apply(_score.fun, axis=1, result_type="expand")

    if _consensus: # if consensus keep only the keys and the method scores
        lr_res = lr_res[_key_cols + [_score.magnitude, _score.specificity]]

    # remove redundant cols for some scores
    if (_score.magnitude is None) | (_score.specificity is None):
        lr_res = lr_res.drop([None], axis=1)

    return lr_res
