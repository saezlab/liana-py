from __future__ import annotations

import anndata
import pandas

from liana.method._pipe_utils import prep_check_adata, assert_covered, filter_resource, \
    filter_reassemble_complexes
from liana.method.ml._ml_utils._filter import filter_ml_resource
from ...resource.ml import select_ml_resource, explode_proddeg
from ...resource import select_resource
from liana.method._pipe_utils._get_mean_perms import _get_means_perms
from liana.method._pipe_utils._aggregate import _aggregate


import pandas as pd
import numpy as np

from functools import reduce




 
def ml_pipe(adata: anndata.AnnData,
               groupby: str,
               resource_name: str,
               resource: pd.DataFrame | None,
               met_est_resource_name: str,
               met_est_resource: pd.DataFrame | None,
               expr_prop: float,
               min_cells: int,
               base: float,
               de_method: str,
               verbose: bool,
               use_raw: bool,
               layer: str | None,
               supp_columns: list | None = None,
               return_all_lrs: bool = False,
               _key_cols: list = None,
               _estimation=None,
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

    # Estimate metabolite abundances
    met_est_result = _run_met_est_method(me_res=met_est_resource, adata=adata, _estimation=_estimation, verbose=verbose)

    # load metabolite-protein resource
    resource = select_resource(resource_name.lower())

    # Filter resource to only include metabolites and genes that were estimated 
    resource = filter_ml_resource(resource, met_est_result.index, adata.var_names)

    # Add relevant columns
    
    # run methods

    return met_est_result.T


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


def _run_met_est_method(me_res: pandas.DataFrame,
                adata: anndata.AnnData,
                _estimation,
                verbose: bool,
                _aggregate_flag: bool = False  # Indicates whether we're generating the consensus
                ) -> pd.DataFrame:
                """
                xyz                
                """
                
                me_res = _estimation.fun(me_res, adata, verbose=verbose)

                return me_res
