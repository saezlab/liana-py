from ..utils import check_mat, check_if_covered, format_vars, filter_resource
from ..resource import select_resource, explode_complexes
from ..utils import reassemble_complexes
from ..scores import get_means_perms

import scanpy as sc
import pandas as pd
import numpy as np


def liana_pipe(adata, groupby, resource_name, resource, de_method,
               n_perms, seed, verbose, _complex_cols, _add_cols,
               _key_cols=None, _score=None):
    if _key_cols is None:
        _key_cols = ['source', 'target', 'ligand_complex', 'receptor_complex']

    # Check and Reformat Mat if needed
    adata.X = check_mat(adata.X, verbose=verbose)

    # Define idents col name
    adata.obs.label = adata.obs[groupby]

    # Remove underscores from gene names
    adata.var_names = format_vars(adata.var_names)

    # get mat mean for SCA
    if 'mat_mean' in _add_cols:  # SHOULD BE METHOD NAME?!?
        adata.uns['mat_mean'] = np.mean(adata.X)

    if resource is None:
        resource = select_resource(resource_name)
    # explode complexes/decomplexify
    resource = explode_complexes(resource)
    # Filter Resource
    resource = filter_resource(resource, adata.var_names)

    # Create Entities
    entities = np.union1d(np.unique(resource["ligand"]), np.unique(resource["receptor"]))

    # Check overlap between resource and adata
    check_if_covered(entities, adata.var_keys, verbose=verbose)

    # Filter to only include the relevant genes
    adata = adata[:, np.intersect1d(entities, adata.var.index)]

    # Get lr results
    lr_res = _get_lr(adata, resource, _key_cols + _complex_cols + _add_cols, de_method)

    # re-assemble complexes
    lr_res = reassemble_complexes(lr_res, _key_cols, _complex_cols, 'min')

    # Calculate Score
    if _score is not None:
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


def _get_lr(adata, resource, relevant_cols, de_method):
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
    # Calc DE stats
    sc.tl.rank_genes_groups(adata, groupby='label', method=de_method)
    # get label cats
    labels = adata.obs.label.cat.categories

    # Method-specific stats
    if 'ligand_zscores' in relevant_cols:  # SHOULD BE METHOD NAME?
        adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X

    dedict = {label: sc.get.rank_genes_groups_df(adata, label).assign(
        label=label).sort_values('names') for label in labels}

    # check if genes are ordered correctly
    if not list(adata.var_names) == list(dedict[labels[0]]['names']):
        raise AssertionError("Variable names did not match DE results!")

    # Calculate Mean, Sum and z-scores by group
    for label in labels:
        temp = adata[adata.obs.label.isin([label])].copy()
        # dedict[label]['sums'] = temp.X.sum(0)
        dedict[label]['means'] = temp.X.mean(0).A.flatten()
        if 'ligand_zscores' in relevant_cols:  # SHOULD BE METHOD NAME?
            dedict[label]['zscores'] = temp.layers['scaled'].mean(0)

    # Get cell identity pairs
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

    # subset to only relevant columns and return
    return lr_res[relevant_cols]
