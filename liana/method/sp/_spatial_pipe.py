import numpy as np
import pandas as pd
from itertools import product
from scipy.sparse import csr_matrix


from liana.utils._utils import _get_props
from liana.method.sp._SpatialMethod import _SpatialMeta, _basis_meta
from liana.method.sp._spatial_utils import _local_to_dataframe, _local_permutation_pvals, _categorize, \
    _simplify_cats, _encode_as_char, _get_ordered_matrix, _standardize_matrix, _rename_means
from liana.method.sp._bivariate_funs import _handle_functions, _global_spatialdm


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
                 n_perm = None, 
                 seed = 1337,
                 nz_threshold=0,
                 remove_self_interactions=True,
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
        
        dist = mdata.obsm[proximity_key]
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
        
        
        if local_fun.__name__== "_local_morans": ## TODO move specifics to method instances
            norm_factor = dist.shape[0] / dist.sum()
            weight = csr_matrix(norm_factor * dist)
        else:
            weight = dist.A
        
        local_score = local_fun(x_mat.T.A, y_mat.T.A, weight)
        mdata.obsm[score_key] = _local_to_dataframe(array=local_score.T,
                                                    idx=mdata.obs.index,
                                                    columns=xy_stats.interaction)
        
        # global scores, TODO should they be be score specific? e.g. spatialMD has its own global score
        if local_fun.__name__== "_local_morans":
            xy_stats.loc[:, ['global_r', 'global_pvals']] = \
                _global_spatialdm(x_mat=_standardize_matrix(x_mat, local=False, axis=1),
                                  y_mat=_standardize_matrix(y_mat, local=False, axis=1),
                                  dist=weight,
                                  seed=seed,
                                  n_perm=n_perm,
                                  pvalue_method='analytical',
                                  positive_only=False
                                  ).T
        else:
            # any other local score
            xy_stats.loc[:,['global_mean','global_sd']] = np.vstack([np.mean(local_score, axis=1), np.std(local_score, axis=1)]).T
            mdata.uns['global_res'] = xy_stats


        if n_perm is not None:
            local_pvals = _local_permutation_pvals(x_mat = x_mat.A.T, 
                                                   y_mat = y_mat.A.T, 
                                                   local_truth=local_score,
                                                   local_fun=local_fun,
                                                   dist=weight, 
                                                   n_perm=n_perm, 
                                                   positive_only=False,
                                                   seed=seed
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





