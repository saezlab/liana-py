from __future__ import annotations

import anndata
import pandas

from liana.method._pipe_utils import prep_check_adata
from liana.method.ml._ml_utils._filter import filter_ml_resource
from ...resource.ml import select_ml_resource, explode_proddeg
from ...resource import select_resource


import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm



def ml_pipe(adata: anndata.AnnData,
               groupby: str,
               output:  str,
               resource_name: str,
               resource: pd.DataFrame | None,
               met_est_resource_name: str,
               met_est_resource: pd.DataFrame | None,
               min_cells: int,
               expr_prop: float,
               verbose: bool,
               seed: int,
               n_perms: int,
               use_raw: bool,
               layer: str | None,
               supp_columns: list | None = None,
               return_all_lrs: bool = False,
               _key_cols: list = None,
               _complex_cols: list = None,
               _score = None
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
    met_est_resource_name
        Name of the resource to be loaded and use for metabolite estimation.
    met_est_resource
        Parameter to enable external resources to be passed. Expects a pandas dataframe 
        with ['HMDB_ID', 'Gene_name'] columns.
    min_cells
        Minimum cells per cell identity
    expr_prop
        Minimum expression proportion for the ligands/receptors (and their subunits) in the
         corresponding cell identities. Set to `0`, to return unfiltered results. 
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
    output
        Flag if full MR scores should be returned or only metabolite estimates.
    supp_columns
        additional columns to be added to the output of each method.
    return_all_lrs
        Bool whether to return all LRs, or only those that surpass the expr_prop threshold.
        `False` by default.
    _key_cols
        columns which make every interaction unique (i.e. PK).
    _score
        Instance of Method classes (None by default - returns LR stats - no methods used).
    _consensus_opts
        Ways to aggregate interactions across methods by default does all aggregations (['Steady',
        'Specificity', 'Magnitude']).

    Returns
    -------
    A adata frame with ligand-receptor results

    """
    # merge relevant columns, check if needed later
    if _key_cols is None:
        _key_cols = ['source', 'target']

    if _score is not None:
        _complex_cols, _add_cols = _score.complex_cols, _score.add_cols
    else:
        _complex_cols = ['ligand_means', 'receptor_means']
        # change to full list and move to _var
        _add_cols = []

    if supp_columns is None:
        supp_columns = []
    _add_cols = _add_cols + ['ligand', 'receptor', 'ligand_props', 'receptor_props'] + supp_columns

    if verbose:
        print(f"Run with {output} as output")

    # Check and Reformat Mat if needed
    adata = prep_check_adata(adata=adata,
                             groupby=groupby,
                             min_cells=min_cells,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose)

    # Load metabolite resource
    met_est_resource = select_ml_resource(met_est_resource_name.lower())

    # Expand resource to one row per producing/degrading enzyme
    met_est_resource = explode_proddeg(met_est_resource)

    if verbose:
        print(f"Estimating abundance of {len(met_est_resource['HMDB_ID'].unique().tolist())} metabolites ")

    # Estimate metabolite abundances, check if ocean etc with flags and if or run_method
    met_est_result = _mebocost_estimation(me_res=met_est_resource, adata=adata, verbose=verbose)

    # Return only metabolite estimates if wanted
    if output == 'ME':
        return met_est_result.T, pd.DataFrame()

    # Whole MR pipeline
    else:
        # load metabolite-protein resource
        resource = select_resource(resource_name.lower())

        # Filter resource to only include metabolites and genes that were estimated 
        resource = filter_ml_resource(resource, met_est_result.index, adata.var_names)

        rel_cols = ['source', 'target', 'ligand', 'receptor', 'ligand_props', 'receptor_props', 'ligand_means', 'receptor_means']

        # Get lr results
        lr_res = _get_lr(adata=adata, resource=resource, met_est=met_est_result)

        # run scoring method
        lr_res = _run_method(lr_res=lr_res, adata=adata, 
                                    return_all_lrs=return_all_lrs,
                                    verbose=verbose, expr_prop=expr_prop,
                                 _score=_score, _key_cols=_key_cols, _complex_cols=_complex_cols,
                                 _add_cols=_add_cols, n_perms=n_perms, seed=seed, met_est_index=met_est_result.index)

        return met_est_result.T, lr_res


def _join_stats(source, target, dedict_gene, dedict_met, resource):
    """
    Joins and renames source-ligand and target-receptor stats to the ligand-receptor resource

    Parameters
    ----------
    source
        Source/Sender cell type
    target
        Target/Receiver cell type
    dedict_gene
        dictionary of gene stats
    dedict_met
        dictionary of metabolite stats
    resource
        Ligand-receptor Resource

    Returns
    -------
    Ligand-Receptor stats

    """
    source_stats = dedict_met[source].copy()
    source_stats.columns = source_stats.columns.map(
        lambda x: 'ligand_' + str(x))
    source_stats = source_stats.rename(
        columns={'ligand_names': 'ligand', 'ligand_label': 'source'})

    target_stats = dedict_gene[target].copy()
    target_stats.columns = target_stats.columns.map(
        lambda x: 'receptor_' + str(x))
    target_stats = target_stats.rename(
        columns={'receptor_names': 'receptor', 'receptor_label': 'target'})

    bound = resource.merge(source_stats).merge(target_stats)

    return bound


def _get_lr(adata, resource, met_est):
    """
    Run DE analysis and merge needed information with resource for LR inference

    Parameters
    ----------
    adata : anndata.AnnData
        adata filtered and formated to contain only the relevant features for lr inference

    resource : pandas.core.frame.DataFrame formatted and filtered resource
    dataframe with the following columns: [interaction, ligand, receptor,
    ligand_complex, receptor_complex]

    met_est : metabolite abundance estimates

    Returns
    -------
    lr_res : pandas.core.frame.DataFrame long-format pandas dataframe with stats
    for all interactions /w matching variables in the dataset.

    """
    # # get label cats
    # labels = adata.obs.label.cat.categories

    # # Sort metabolite estimates by index
    # met_est = met_est.sort_index()

    # # initialize dict
    # dedict_met = {}
    # for label in labels:
    #     a = _get_props(scipy.sparse.csr_matrix(met_est.values.T))
    #     stats = pd.DataFrame({'names': met_est.index, 'props': a}). \
    #         assign(label=label).sort_values('names')
    #     dedict_met[label] = stats

    # # check if genes are ordered correctly
    # if not list(met_est.index) == list(dedict_met[labels[0]]['names']):
    #     raise AssertionError("Variable names did not match DE results!")

    # # Calculate Mean, logFC and z-scores by group
    # for label in labels:
    #     temp = adata[adata.obs.label.isin([label])]
    #     dedict_met[label]['means'] = scipy.sparse.csr_matrix(met_est.values.T).mean(axis=0).A.flatten()

    # # initialize dict
    # dedict_gene = {}

    # for label in labels:
    #     temp = adata[adata.obs.label == label, :]
    #     a = _get_props(temp.X)
    #     stats = pd.DataFrame({'names': temp.var_names, 'props': a}). \
    #         assign(label=label).sort_values('names')
    #     dedict_gene[label] = stats

    # # check if genes are ordered correctly
    # if not list(adata.var_names) == list(dedict_gene[labels[0]]['names']):
    #     raise AssertionError("Variable names did not match DE results!")

    # # Calculate Mean, logFC and z-scores by group
    # for label in labels:
    #     temp = adata[adata.obs.label.isin([label])]
    #     dedict_gene[label]['means'] = temp.X.mean(axis=0).A.flatten()


    # # Create df /w cell identity pairs
    # pairs = (pd.DataFrame(np.array(np.meshgrid(labels, labels))
    #                       .reshape(2, np.size(labels) * np.size(labels)).T)
    #          .rename(columns={0: "source", 1: "target"}))

    # # Join Stats
    # lr_res = pd.concat(
    #     [_join_stats(source, target, dedict_gene, dedict_met, resource) for source, target in
    #      zip(pairs['source'], pairs['target'])]
    # )

    # return  lr_res  


    labels = adata.obs.label.cat.categories
    met_est = met_est.sort_index()

    dedict_met = {label: pd.DataFrame({
        'names': met_est.index,
        'props': _get_props(scipy.sparse.csr_matrix(met_est.values.T)),
        'label': label
    }).sort_values('names') for label in labels}

    if list(met_est.index) != list(dedict_met[labels[0]]['names']):
        raise AssertionError("Variable names did not match DE results!")

    for label in labels:
        dedict_met[label]['means'] = scipy.sparse.csr_matrix(met_est.values.T).mean(axis=0).A.flatten()

    dedict_gene = {label: pd.DataFrame({
        'names': adata[adata.obs.label == label].var_names,
        'props': _get_props(adata[adata.obs.label == label].X),
        'label': label
    }).sort_values('names') for label in labels}

    if list(adata.var_names) != list(dedict_gene[labels[0]]['names']):
        raise AssertionError("Variable names did not match DE results!")

    for label in labels:
        dedict_gene[label]['means'] = adata[adata.obs.label == label].X.mean(axis=0).A.flatten()

    pairs = pd.DataFrame(np.array(np.meshgrid(labels, labels)).reshape(2, len(labels) ** 2).T, columns=["source", "target"])

    lr_res = pd.concat([_join_stats(source, target, dedict_gene, dedict_met, resource) for source, target in pairs.to_numpy()])

    return lr_res


# Function to get gene expr proportions
def _get_props(X_mask):
    return X_mask.getnnz(axis=0) / X_mask.shape[0]


def _mebocost_estimation(me_res, adata, verbose) -> pd.DataFrame: 
    """
    Estimate metabolite abundances using mebocost flavor

    Parameters
    ----------
    me_res : pandas.core.frame.DataFrame
        metabolite-gene associations
    
    adata : anndata.AnnData
        object with gene expression data

    verbose : bool
        verbosity

    Returns
    -------
    met_est : pandas.core.frame.DataFrame
        metabolite abundance estimates

    """

    method = 'mean' # in mebocost there are more options, build_in
    met_gene = me_res
    mIdList = met_gene['HMDB_ID'].unique().tolist()

    with_exp_gene_m = []
    met_from_gene = pd.DataFrame()
    for mId in mIdList:
        gene_pos = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'product')]['gene_name'].tolist()
        gene_pos = set(gene_pos) & set(adata.var_names.tolist())
        gene_neg = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'substrate')]['gene_name'].tolist()
        gene_neg = set(gene_neg) & set(adata.var_names.tolist())

        if len(gene_pos) == 0:
            continue

        with_exp_gene_m.append(mId)
        pos_g_index = np.where(adata.var_names.isin(gene_pos))
        pos_exp = pd.DataFrame(adata.T[pos_g_index].X.toarray(), 
                            index = adata.var_names[pos_g_index].tolist(),
                            columns = adata.obs_names.tolist())

        if not gene_neg:
            m_from_enzyme = pos_exp.agg(method)
        else:
            neg_g_index = np.where(adata.var_names.isin(gene_neg))
            neg_exp = pd.DataFrame(adata.T[neg_g_index].X.toarray(), 
                            index = adata.var_names[neg_g_index].tolist(),
                            columns = adata.obs_names.tolist())
            pos = pos_exp.agg(method)
            neg = neg_exp.agg(method)
            m_from_enzyme = pos - neg
        met_from_gene = pd.concat([met_from_gene, m_from_enzyme], axis = 1)

    met_from_gene.columns = with_exp_gene_m
    met_from_gene = met_from_gene.T

    if verbose:
        print('Metabolites with gene expression: ', len(with_exp_gene_m))
        print('Metabolites without gene expression: ', len(mIdList) - len(with_exp_gene_m))

    return met_from_gene

def _run_method(lr_res: pandas.DataFrame,
                adata: anndata.AnnData,
                expr_prop: float, # build in to mebocost
                _score,
                _key_cols: list,
                _complex_cols: list,
                _add_cols: list,
                n_perms: int,
                seed: int,
                return_all_lrs: bool,
                verbose: bool,
                _aggregate_flag: bool = False,  # Indicates whether we're generating the consensus
                met_est_index: list = None,
                ) -> pd.DataFrame:

    _add_cols = _add_cols + ['ligand', 'receptor']

    agg_fun = np.mean
    norm_factor = None

    if _score.permute:
        perms, ligand_pos, receptor_pos, labels_pos = \
            _get_means_perms(adata=adata,
                             lr_res=lr_res,
                             n_perms=n_perms,
                             seed=seed,
                             agg_fun=agg_fun,
                             norm_factor=norm_factor,
                             verbose=verbose,
                             met_est_index=met_est_index,)
        lr_res[[_score.magnitude, _score.specificity]] = \
            lr_res.apply(_score.fun, axis=1, result_type="expand",
                         perms=perms, ligand_pos=ligand_pos,
                         receptor_pos=receptor_pos, labels_pos=labels_pos)
    else:  # non-perm funs
        lr_res[[_score.magnitude, _score.specificity]] = \
            lr_res.apply(_score.fun, axis=1, result_type="expand")

    # if return_all_lrs:
    #     # re-append rest of results
    #     lr_res = pd.concat([lr_res, rest_res], copy=False)
    #     if _score.magnitude is not None:
    #         fill_value = _assign_min_or_max(lr_res[_score.magnitude],
    #                                         _score.magnitude_ascending)
    #         lr_res.loc[~lr_res['lrs_to_keep'], _score.magnitude] = fill_value
    #     if _score.specificity is not None:
    #         fill_value = _assign_min_or_max(lr_res[_score.specificity],
    #                                         _score.specificity_ascending)
    #         lr_res.loc[~lr_res['lrs_to_keep'], _score.specificity] = fill_value

    if _aggregate_flag:  # if consensus keep only the keys and the method scores
        lr_res = lr_res[_key_cols + [_score.magnitude, _score.specificity]]

    # remove redundant cols for some scores
    if (_score.magnitude is None) | (_score.specificity is None):
        lr_res = lr_res.drop([None], axis=1)

    return lr_res    


def _get_means_perms(adata: anndata.AnnData,
                     lr_res: pandas.DataFrame,
                     n_perms: int,
                     seed: int,
                     agg_fun,
                     norm_factor: float | None,
                     met_est_index: pandas.Index,
                     verbose: bool):
    """
    Generate permutations and indices required for permutation-based methods

    Parameters
    ----------
    adata
        Annotated data matrix.
    lr_res
        Ligand-receptor stats DataFrame
    n_perms
        Number of permutations to be calculated
    seed
        Random seed for reproducibility.
    agg_fun
        function by which to aggregate the matrix, should take `axis` argument
    norm_factor
        additionally normalize the data by some factor (e.g. matrix max for CellChat)
    met_est_index
        Index of metabolites that were estimated
    verbose
        Verbosity bool

    Returns
    -------
    Tuple with:
        - perms: 3D tensor with permuted averages per cluster
        - ligand_pos: Index of the ligand in the tensor
        - receptor_pos: Index of the receptor in the perms tensor
        - labels_pos: Index of cell identities in the perms tensor
    """

    # initialize rng
    rng = np.random.default_rng(seed=seed)

    if isinstance(norm_factor, np.float):
        adata.X /= norm_factor

    # define labels and dict
    labels = adata.obs.label.cat.categories
    labels_dict = {label: adata.obs.label.isin([label]) for label in labels}

    # indexes to be shuffled
    idx = np.arange(adata.X.shape[0])

    # Perm should be a cube /w dims: n_perms x idents x n_genes
    perms = np.zeros((n_perms, labels.shape[0], adata.shape[1]))

    # Assign permuted matrix
    for perm in tqdm(range(n_perms), disable=not verbose):
        perm_idx = rng.permutation(idx)
        perm_mat = adata.X[perm_idx]
        # populate matrix /w permuted means
        for cind in range(labels.shape[0]):
            perms[perm, cind] = agg_fun(perm_mat[labels_dict[labels[cind]]], axis=0)

    # Get indexes for each gene and label in the permutations
    ligand_pos = {entity: np.where(met_est_index == entity)[0][0] for entity
                  in lr_res['ligand']}
    receptor_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity
                    in lr_res['receptor']}
    labels_pos = {labels[pos]: pos for pos in range(labels.shape[0])}

    return perms, ligand_pos, receptor_pos, labels_pos