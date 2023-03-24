from __future__ import annotations

from liana.method._pipe_utils import prep_check_adata
from liana.method.ml._ml_utils._filter import filter_ml_resource
from ...resource.ml import select_ml_resource
from ...resource import select_resource
from .estimations import _metalinks_estimation

from anndata import AnnData
from pandas import DataFrame, Index, concat
import numpy as np
from scipy import sparse
from tqdm import tqdm
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.distributions.empirical_distribution import ECDF



def ml_pipe(adata: AnnData,
               groupby: str,
               resource_name: str,
               resource: DataFrame | None,
               met_est_resource_name: str,
               met_est_resource: DataFrame | None,
               est_fun: str,
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

    # Check and Reformat Mat if needed
    adata = prep_check_adata(adata=adata,
                             groupby=groupby,
                             min_cells=min_cells,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose)

    # Load metabolite resource
    met_est_resource = select_ml_resource(met_est_resource_name)

    # Estimate metabolite abundances, check if ocean etc with flags and if or run_method
    met_est_result = _metalinks_estimation(me_res=met_est_resource, 
                                            adata=adata, 
                                            est_fun = est_fun,
                                            verbose=verbose)

    # met_est_result = met_est_result.sort_index()

    adata.obsm['metabolite_abundance'] = met_est_result[0] ################ attention
    
    adata.uns['met_index'] = met_est_result[1]

    mask = DataFrame(met_est_result[2].todense(), columns=adata.var_names, index=met_est_result[1])

    # PD_genes = _save_PD_names(met_est_result[1], met_est_resource)

    # load metabolite-protein resource
    resource = select_resource(resource_name.lower())

    # Filter resource to only include metabolites and genes that were estimated 
    resource = filter_ml_resource(resource, adata.uns['met_index'], adata.var_names)

    if verbose:
        print(f"Running ligand-receptor inference on {len(resource['ligand'].unique().tolist())} unique ligands "
                f"and {len(resource['receptor'].unique().tolist())} unique receptors ")
                
    # Get lr results
    lr_res = _get_lr(adata=adata, resource=resource, expr_prop=expr_prop)

    # run scoring method
    lr_res = _run_method(lr_res=lr_res, adata=adata, 
                                return_all_lrs=return_all_lrs,
                                verbose=verbose, expr_prop=expr_prop,
                                _score=_score, _key_cols=_key_cols, _complex_cols=_complex_cols,
                                _add_cols=_add_cols, n_perms=n_perms, seed=seed)

    # correct pvalues for fdr 
    #lr_res[_score.specificity] = fdrcorrection(lr_res[_score.specificity])[1]

    return lr_res, met_est_result[0], mask.T



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


def _get_lr(adata, resource, expr_prop):
    """
    Run DE analysis and merge needed information with resource for LR inference

    Parameters
    ----------
    adata : anndata.AnnData
        adata filtered and formated to contain only the relevant features for lr inference

    resource : pandas.core.frame.DataFrame formatted and filtered resource
    dataframe with the following columns: [interaction, ligand, receptor,
    ligand_complex, receptor_complex]

    Returns
    -------
    lr_res : pandas.core.frame.DataFrame long-format pandas dataframe with stats
    for all interactions /w matching variables in the dataset.

    """


    labels = adata.obs.label.cat.categories

    dedict_met = {label: DataFrame({
        'names': adata.uns['met_index'],
        'props': _get_props(adata.obsm['metabolite_abundance']),
        'label': label
    }).sort_values('names') for label in labels}

    if list(adata.uns['met_index']) != list(dedict_met[labels[0]]['names']):
        raise AssertionError("Variable names did not match DE results!")

    for label in labels:
        dedict_met[label]['means'] = adata[adata.obs.label == label].obsm['metabolite_abundance'].mean(axis=0).A.flatten()

    dedict_gene = {label: DataFrame({
        'names': adata[adata.obs.label == label].var_names,
        'props': _get_props(adata[adata.obs.label == label].X),
        'label': label
    }).sort_values('names') for label in labels}

    # if list(adata.var_names) != list(dedict_gene[labels[0]]['names']):
    #     raise AssertionError("Variable names did not match DE results!")

    for label in labels:
        dedict_gene[label]['means'] = adata[adata.obs.label == label].X.mean(axis=0).A.flatten()

    pairs = DataFrame(np.array(np.meshgrid(labels, labels)).reshape(2, len(labels) ** 2).T, columns=["source", "target"])

    lr_res = concat([_join_stats(source, target, dedict_gene, dedict_met, resource) for source, target in pairs.to_numpy()])

    lr_res.drop_duplicates(inplace=True)

    lr_res = lr_res[(lr_res['receptor_props'] >= expr_prop) & (lr_res['ligand_means'] > 0) & (lr_res['ligand_props'] >= expr_prop)]

    return lr_res


# Function to get gene expr proportions
def _get_props(X_mask):
    return X_mask.getnnz(axis=0) / X_mask.shape[0]


def _run_method(lr_res: DataFrame,
                adata: AnnData,
                expr_prop: float, # build in to metalinks
                _score,
                _key_cols: list,
                _complex_cols: list,
                _add_cols: list,
                n_perms: int,
                seed: int,
                return_all_lrs: bool,
                verbose: bool,
                _aggregate_flag: bool = False,  # Indicates whether we're generating the consensus
                ) -> DataFrame:

    _add_cols = _add_cols + ['ligand', 'receptor']

    agg_fun = np.mean
    norm_factor = None

    tqdm.pandas()

    if _score.permute:
        perms, ligand_pos, receptor_pos, labels_pos, perms2 = \
            _get_means_perms(adata=adata,
                             lr_res=lr_res,
                             n_perms=n_perms,
                             seed=seed,
                             agg_fun=agg_fun,
                             norm_factor=norm_factor,
                             verbose=verbose)

        if verbose:
            print("Permutations done, calculating scores...")

        lr_res[[_score.magnitude, _score.specificity]] = \
            lr_res.progress_apply(_score.fun, axis=1, result_type="expand",
                         perms=perms, ligand_pos=ligand_pos,
                         receptor_pos=receptor_pos, labels_pos=labels_pos, perms2=perms2)
    else:  # non-perm funs
        lr_res[[_score.magnitude, _score.specificity]] = \
            lr_res.apply(_score.fun, axis=1, result_type="expand")


    if _aggregate_flag:  # if consensus keep only the keys and the method scores
        lr_res = lr_res[_key_cols + [_score.magnitude, _score.specificity]]

    # remove redundant cols for some scores
    if (_score.magnitude is None) | (_score.specificity is None):
        lr_res = lr_res.drop([None], axis=1)

    return lr_res    


def _get_means_perms(adata: AnnData,
                     lr_res: DataFrame,
                     n_perms: int,
                     seed: int,
                     agg_fun,
                     norm_factor: float | None,
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
    perms2 = np.zeros((n_perms, labels.shape[0], adata.obsm['metabolite_abundance'].shape[1]))


    # Assign permuted matrix
    for perm in tqdm(range(n_perms), disable=not verbose):
        perm_idx = rng.permutation(idx)
        perm_mat = adata.X[perm_idx]
        perm_mat2 = adata.obsm['metabolite_abundance'][perm_idx]
        # populate matrix /w permuted means
        for cind in range(labels.shape[0]):
            perms[perm, cind] = agg_fun(perm_mat[labels_dict[labels[cind]]], axis=0)
            perms2[perm, cind] = agg_fun(perm_mat2[labels_dict[labels[cind]]], axis=0)



    # Get indexes for each gene and label in the permutations
    ligand_pos = {entity: np.where(adata.uns['met_index'] == entity)[0][0] for entity
                  in lr_res['ligand']}
    receptor_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity
                    in lr_res['receptor']}
    labels_pos = {labels[pos]: pos for pos in range(labels.shape[0])}

    return perms, ligand_pos, receptor_pos, labels_pos, perms2



def _get_lr_pvals(x, perms, ligand_pos, receptor_pos, labels_pos, perms2, agg_fun,
                  ligand_col='ligand_means', receptor_col='receptor_means'):
    """
    Calculate Permutation means and p-values

    Parameters
    ----------
    x
        DataFrame row
    perms
        3D tensor with permuted averages per cluster
    ligand_pos
        Index of the ligand in the tensor
    receptor_pos
        Index of the receptor in the perms tensor
    labels_pos
        Index of cell identities in the perms tensor
    agg_fun
        function to aggregate the ligand and receptor

    Returns
    -------
    A tuple with lr_score (aggregated according to `agg_fun`) and ECDF p-value for x

    """
    # actual lr_score
    lr_score = agg_fun(x[ligand_col], x[receptor_col])

    if lr_score == 0:
        return 0, 1

    # Permutations lr mean
    ligand_perm_means = perms2[:, labels_pos[x.source], ligand_pos[x.ligand]]
    receptor_perm_means = perms[:, labels_pos[x.target], receptor_pos[x.receptor]]
    lr_perm_score = agg_fun(ligand_perm_means, receptor_perm_means)

    p_value = (1 - ECDF(lr_perm_score)(lr_score))

    return lr_score, p_value



# # write a function that creates an array with the corresponding gene names in the resource for the metabolite names in the index
# def _save_PD_names(index, resource):
#     # create array with three columns: metabolite name, producing genes, degrading genes
#     df = DataFrame(index, columns=['metabolite'])
#     df['producing_genes'] = 'no values'
#     df['degrading_genes'] = 'no values'
#     # for every metabolite in index, find the row in the resource that match the metabolite name and store the gene names of producing and degrading genes
#     for i in range(len(index)):
#         a = resource[resource['HMDB'] == index[i]]
#         # df['producing_genes'][i] = a['GENE'][a['direction'] == 'producing'].values.copy()
#         # df['degrading_genes'][i] = a['GENE'][a['direction'] == 'degrading'].values.copy()

#         df.at[i, 'producing_genes'] = a['GENE'][a['direction'] == 'producing'].values
#         df.at[i, 'degrading_genes'] = a['GENE'][a['direction'] == 'degrading'].values



#     return df




   

  


