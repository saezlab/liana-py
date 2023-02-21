import numpy as np
import pandas as pd
from itertools import product
from scipy.sparse import csr_matrix


from liana.utils._utils import _get_props
from liana.method.sp._SpatialMethod import _SpatialMeta, _basis_meta
from liana.method.sp._spatial_utils import _local_to_dataframe, _categorize, \
    _simplify_cats, _encode_as_char, _get_ordered_matrix, _rename_means, _get_local_scores, _get_global_scores, _proximity_to_weight
from liana.method.sp._bivariate_funs import _handle_functions


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
                 proximity_key,
                 score_key = "local_score",
                 categorize = False,
                 pvalue_method : (str | None) = 'permutation',
                 n_perms: int = 50,
                 seed = 1337,
                 nz_threshold=0,
                 remove_self_interactions=True,
                 positive_only=False, ## TODO change to categorical
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
        
        proximity = mdata.obsm[proximity_key]
        local_fun = _handle_functions(function_name)
        weight = _proximity_to_weight(proximity, local_fun)
        
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
        
        if remove_self_interactions:
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
            
        
        local_scores, local_pvals = _get_local_scores(x_mat = x_mat.T,
                                                      y_mat = y_mat.T,
                                                      local_fun = local_fun,
                                                      weight = weight,
                                                      seed = seed,
                                                      n_perms = n_perms,
                                                      pvalue_method = pvalue_method,
                                                      positive_only=positive_only,
                                                      )
        
        mdata.obsm[score_key] = _local_to_dataframe(array=local_scores.T,
                                                    idx=xdata.obs.index,
                                                    columns=xy_stats.interaction)
        mdata.obsm['local_pvals'] = _local_to_dataframe(array=local_pvals.T,
                                                        idx=ydata.obs.index,
                                                        columns=xy_stats.interaction)
        
        # global scores fun
        xy_stats = _get_global_scores(xy_stats=xy_stats,
                                      x_mat=x_mat,
                                      y_mat=y_mat,
                                      local_fun=local_fun,
                                      pvalue_method=pvalue_method,
                                      weight=weight,
                                      seed=seed,
                                      n_perms=n_perms,
                                      positive_only=positive_only,
                                      local_scores=local_scores,
                                      )
            
        # save to uns
        mdata.uns['global_res'] = xy_stats

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
    adata.X = csr_matrix(adata.X) ## TODO change to ~prep_check_adata (but not for gene expression alone)
    
    global_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'non_zero': _get_props(adata.X)},
                                index=adata.var_names)
    global_stats = global_stats.reset_index().rename(columns={'index': 'entity'}).rename(columns={'interaction': 'entity'})
    global_stats = global_stats[global_stats['non_zero'] >= nz_thr]

    return global_stats
