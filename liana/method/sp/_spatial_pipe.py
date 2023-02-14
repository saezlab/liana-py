import numpy as np
import pandas as pd
from itertools import product
from scipy.sparse import csr_matrix


from liana.resource import select_resource
from liana.method._pipe_utils._reassemble_complexes import explode_complexes
from liana.method._pipe_utils import prep_check_adata, filter_resource, assert_covered, filter_reassemble_complexes
from liana.utils._utils import _get_props
from liana.method.sp._SpatialMethod import _SpatialMeta, _basis_meta
from liana.method.sp._spatial_utils import _local_to_dataframe, _local_permutation_pvals, _categorize, _simplify_cats, _encode_as_char, _handle_functions



class SpatialBivariate(_SpatialMeta):
    def __init__(self, _method):
        super().__init__(method_name=_method.method_name,
                         key_cols=_method.key_cols,
                         reference=_method.reference,
                         )
        self._method = _method

    def __call__(self,
                 mdata, 
                 function_name,
                 x_mod,
                 y_mod, 
                 dist,
                 score_key = "local_score",
                 categorize = False,
                 n_perm = None, 
                 nz_threshold=0,
                 keep_self_interactions=False,
                 ):
        """
        Global Bivariate analysis pipeline
        
        Parameters
        ----------
        mdata : anndata
            The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and
        x_mod : str
            Name of the modality to use as x
        y_mod : str
            Name of the modality to use as y
        nz_threshold : float
            Threshold for the proportion of non-zero values in a cell for the interaction to be considered
            
        Returns
        -------
        Returns xy_stats, x_pos, y_pos
        
        """
        
        # TODO currently works only with .X
        xdata = mdata[x_mod]
        ydata = mdata[y_mod]
        
        
        local_fun = _handle_functions(function_name)
        
        # change Index names to entity
        xdata.var_names.rename('entity', inplace=True)
        ydata.var_names.rename('entity', inplace=True)
        
        
        x_stats = _rename_means(_anndata_to_stats(xdata, nz_threshold), entity='x')
        y_stats = _rename_means(_anndata_to_stats(ydata, nz_threshold), entity='y')
        
        xy_stats = pd.DataFrame(list(product(x_stats['x_entity'], 
                                            y_stats['y_entity'])
                                    ),
                                columns=['x_entity', 'y_entity'])
        
        # join global stats to LRs from resource
        xy_stats = xy_stats.merge(x_stats).merge(y_stats)
        
        xy_stats['interaction'] = xy_stats['x_entity'] + '&' + xy_stats['y_entity']
        
        # Remove self-interactions (when x_mod = y_mod)
        xy_stats = xy_stats[xy_stats['x_entity'] != xy_stats['y_entity']]
        
        # assign the positions of x, y to the adata
        x_pos = {entity: np.where(xdata.var_names == entity)[0][0] for entity in xy_stats['x_entity']}
        y_pos = {entity: np.where(ydata.var_names == entity)[0][0] for entity in xy_stats['y_entity']}
        # reorder columns
        xy_stats = xy_stats.reindex(columns=sorted(xy_stats.columns))
        
        # convert to spot_n x xy_n matrices
        x_mat = _get_ordered_matrix(mat=mdata[x_mod].X,
                                    pos=x_pos,
                                    order=xy_stats['x_entity'])
        y_mat = _get_ordered_matrix(mat=mdata[y_mod].X,
                                    pos=y_pos,
                                    order=xy_stats['y_entity'])
        
        
        if local_fun.__name__== "_local_morans": ## TODO move specifics to method instances
            norm_factor = dist.shape[0] / dist.sum()
            weight = csr_matrix(norm_factor * dist)
        else:
            weight = dist.A
        
        local_score = local_fun(x_mat.T.A, y_mat.T.A, weight)
        mdata.obsm[score_key] = _local_to_dataframe(array=local_score.T,
                                                    idx=mdata.obs.index,
                                                    columns=xy_stats.interaction)
        
        # global scores, TODO should be score specific, e.g. spatialMD has its own global score
        if local_fun.__name__== "_local_morans":
            x
        else:
            xy_stats.loc[:,['local_mean','local_sd']] = np.vstack([np.mean(local_score, axis=1), np.std(local_score, axis=1)]).T
            mdata.uns['global_res'] = xy_stats


        if n_perm is not None:
            local_pvals = _local_permutation_pvals(x_mat = x_mat.A.T, 
                                                   y_mat = y_mat.A.T, 
                                                   local_truth=local_score,
                                                   local_fun=local_fun,
                                                   dist=weight, 
                                                   n_perm=n_perm, 
                                                   positive_only=False,
                                                   seed=0
                                                   )
            mdata.obsm['local_pvals'] = _local_to_dataframe(array=local_pvals.T,
                                                            idx=mdata.obs.index,
                                                            columns=xy_stats.interaction)
        
        if categorize:
            # TODO categorizing is currently done following standardization of the matrix
            # i.e. each variable is standardized independently, and then a category is
            # defined based on the values within each stop.
            # (taking into consideration when the input is signed, but not weight or surrounding spots).
            local_catageries = _categorize(_encode_as_char(x_mat.A), _encode_as_char(y_mat.A))
            ## TODO these to helper function that can extract multiple of these
            # and these all saved as arrays, or alternatively saved a modalities in mudata
            mdata.obsm['local_categories'] = _local_to_dataframe(array=local_catageries.T,
                                                                idx=mdata.obs.index,
                                                                columns=xy_stats.interaction)
            mdata.obsm['local_categories'] = _simplify_cats(mdata.obsm['local_categories'])
            
        return mdata


basis = SpatialBivariate(_basis_meta)



def _anndata_to_stats(adata, nz_thr=0.1):
    from scipy.sparse import csr_matrix
    adata.X = csr_matrix(adata.X) ## TODO change to ~prep_check_adata (but not for gene expression alone)
    
    global_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'non_zero': _get_props(adata.X)},
                                index=adata.var_names)
    global_stats = global_stats.reset_index().rename(columns={'index': 'entity'}).rename(columns={'interaction': 'entity'})
    global_stats = global_stats[global_stats['non_zero'] >= nz_thr]

    return global_stats


### TODO into another file & complete pipe (as liana in sc)
def _global_lr_pipe(adata,
                    resource,
                    resource_name,
                    expr_prop,
                    use_raw,
                    layer,
                    verbose,
                    _key_cols,
                    _complex_cols,
                    _obms_keys,
                    ):
    """
    Global Spatial ligand-receptor analysis pipeline

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and
    resource
        Ligand-receptor resource as a pandas dataframe with columns: ligand, receptor
    resource_name
        Name of the resource
    expr_prop
        Proportion of expression of the ligand and receptor in a cell for the interaction to be considered
    layer
        Layer to use for the analysis. If None, use raw or X
    verbose
        Verbosity
    _obms_keys
        Keys of the adata.obsm to use for the analysis
    _key_cols
        Columns to use as keys for the resource
    _complex_cols
        Columns to use as keys for the complexes

    Returns
    -------
    Returns adata, lr_res, ligand_pos, receptor_pos
    

    """
    # prep adata
    adata = prep_check_adata(adata=adata,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose,
                             groupby=None,
                             min_cells=None,
                             obsm_keys=_obms_keys
                             )

    # select & process resource
    if resource is None:
        resource = select_resource(resource_name.lower())
    resource = explode_complexes(resource)
    resource = filter_resource(resource, adata.var_names)

    # get entities
    entities = np.union1d(np.unique(resource["ligand"]),
                          np.unique(resource["receptor"]))

    # Check overlap between resource and adata  TODO check if this works
    assert_covered(entities, adata.var_names, verbose=verbose)

    # Filter to only include the relevant features
    adata = adata[:, np.intersect1d(entities, adata.var.index)]

    # global_stats
    global_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'props': _get_props(adata.X)},
                                index=adata.var_names).reset_index().rename(
        columns={'index': 'gene'})

    # join global stats to LRs from resource
    lr_res = resource.merge(_rename_means(global_stats, entity='ligand')).merge(
        _rename_means(global_stats, entity='receptor'))

    # get lr_res /w relevant x,y (lig, rec) and filter acc to expr_prop
    lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                         expr_prop=expr_prop,
                                         _key_cols=_key_cols,
                                         complex_cols=_complex_cols
                                         )

    # assign the positions of x, y to the adata
    ligand_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity in
                  lr_res['ligand']}
    receptor_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity in
                    lr_res['receptor']}

    return adata, lr_res, ligand_pos, receptor_pos



def _rename_means(lr_stats, entity):
    df = lr_stats.copy()
    df.columns = df.columns.map(lambda x: entity + '_' + str(x) if x != 'gene' else 'gene')
    return df.rename(columns={'gene': entity})


def _get_ordered_matrix(mat, pos, order):
    _indx = np.array([pos[x] for x in order])
    return mat[:, _indx].T


