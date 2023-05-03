from __future__ import annotations

from .._pipe_utils import prep_check_adata
from .._pipe_utils._get_mean_perms import _get_means_perms, _get_mat_idx
from ._ml_utils._filter import filter_ml_resource
from ...resource.ml import select_ml_resource
from ...resource import select_resource
from .estimations import _metalinks_estimation
from .._liana_pipe import _join_stats
from .._pipe_utils._pre import _get_props

from anndata import AnnData
from pandas import DataFrame, Index, concat
import numpy as np
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
            est_only: bool = False,
            pass_mask: bool = False,
            correct_fdr: bool = False,
            _key_cols: list = None,
            _complex_cols: list = None,
            _score = None, 
            **kwargs
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
    est_fun
        Estimation function to be used
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
    est_only
        Bool whether to return only the estimated metabolite abundances and not run the LR inference.
        `False` by default.
    pass_mask
        Bool whether to pass the mask to the estimation function.
        `False` by default.
    correct_fdr
        Bool whether to correct for multiple testing using FDR.
    _key_cols
        columns which make every interaction unique (i.e. PK).
    _score
        Instance of Method classes (None by default - returns LR stats - no methods used).
    _complex_cols
        Columns to be used for complex aggregation (['ligand_means', 'receptor_means'] by default)


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
                                            verbose=verbose, 
                                            pass_mask=pass_mask, 
                                            **kwargs)

    #assign results to adata
    adata.obsm['metabolite_abundance'] = met_est_result[0]
    adata.uns['met_index'] = met_est_result[1]

    # allow early exit e.g. for metabolite estimation benchmarking
    if est_only:
        return adata.obsm['metabolite_abundance'], adata.uns['met_index']
    
    # # save mask for gene plots
    # if pass_mask:
    mask = DataFrame(met_est_result[2].todense(), columns=adata.var_names, index=met_est_result[1])

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

    # correct for multiple testing
    if correct_fdr:
        lr_res[_score.specificity] = fdrcorrection(lr_res[_score.specificity])[1]

    return lr_res, met_est_result[0], mask.T



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

    # get label cats    
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

    if list(adata.var_names) != list(dedict_gene[labels[0]]['names']):
        raise AssertionError("Variable names did not match DE results!")

    for label in labels:
        dedict_gene[label]['means'] = adata[adata.obs.label == label].X.mean(axis=0).A.flatten()

    pairs = DataFrame(np.array(np.meshgrid(labels, labels)).reshape(2, len(labels) ** 2).T, columns=["source", "target"])

    lr_res = concat([_join_stats(source, target, resource, dedict_gene, dedict_met) for source, target in pairs.to_numpy()])

    lr_res.drop_duplicates(inplace=True)

    lr_res = lr_res[(lr_res['receptor_props'] >= expr_prop) & (lr_res['ligand_means'] > 0) & (lr_res['ligand_props'] >= expr_prop)]

    return lr_res



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
    
        perms = _get_means_perms(adata=adata,
                            n_perms=n_perms,
                            seed=seed,
                            agg_fun=agg_fun,
                            norm_factor=norm_factor,
                            verbose=verbose, 
                            met = True)
        
        perms_ligand = perms[0]
        perms_receptor = perms[1]
        
        # get tensor indexes for ligand, receptor, source, target
        ligand_idx, receptor_idx, source_idx, target_idx = _get_mat_idx(adata, lr_res, met = True)
        
        # ligand and receptor perms
        ligand_stat_perms = perms_ligand[:, source_idx, ligand_idx]
        receptor_stat_perms = perms_receptor[:, target_idx, receptor_idx]
        # stack them together
        perm_stats = np.stack((ligand_stat_perms, receptor_stat_perms), axis=0)

        if verbose:
            print("Permutations done, calculating scores...")

        scores = _score.fun(x=lr_res,
                            perm_stats=perm_stats)
        
    else:  # non-perm funs
        
        scores = _score.fun(x=lr_res)
            
    lr_res.loc[:, _score.magnitude] = scores[0]
    lr_res.loc[:, _score.specificity] = scores[1]


    if _aggregate_flag:  # if consensus keep only the keys and the method scores
        lr_res = lr_res[_key_cols + [_score.magnitude, _score.specificity]]

    # remove redundant cols for some scores
    if (_score.magnitude is None) | (_score.specificity is None):
        lr_res = lr_res.drop([None], axis=1)

    return lr_res    







   

  


