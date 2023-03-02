import numpy as np
import pandas as pd
from itertools import product
from scipy.sparse import csr_matrix


from liana.utils._utils import _get_props, obsm_to_adata
from liana.method._pipe_utils._pre import _choose_mtx_rep

from liana.method.sp._SpatialMethod import _SpatialMeta, _basis_meta

from liana.method.sp._spatial_utils import _local_to_dataframe, _categorize, \
    _simplify_cats, _encode_as_char, _get_ordered_matrix, _rename_means, _run_scores_pipeline, \
        _proximity_to_weight, _handle_proximity
    
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
                 proximity_key = 'proximity',
                 mod_added = "local_scores",
                 add_categories = False,
                 pvalue_method: (str | None) = None,
                 positive_only=False, ## TODO change to categorical
                 n_perms: int = 50,
                 seed = 1337,
                 nz_threshold=0,
                 remove_self_interactions=True,
                 proximity = None,
                 x_use_raw=False,
                 x_layer = None,
                 y_use_raw=False,
                 y_layer = None,
                 inplace = True,
                 ):
        """
        Global Bivariate analysis pipeline
        
        Parameters
        ----------
        mdata : anndata
            MuData object containing two modalities of interest
        function_name : str
            Name of the function to use for the local analysis.
        x_mod : str
            Name of the modality to use as x
        y_mod : str
            Name of the modality to use as y
        proximity_key : str
            Key to use to retrieve the proximity matrix from adata.obsp.
        mod_added : str
            Name of the modality to add to the MuData object (in case of inplace=True)
        add_categories : bool
            Whether to add_categories the local scores or not
        pvalue_method : str
            Method to obtain P-values: One out of ['permutation', 'analytical', None];
        positive_only : bool
            Whether to only consider positive local scores
        n_perms : int
            Number of permutations to use for the p-value calculation (when set to permutation)
        seed : int
            Seed to use for the permutation
        nz_threshold : int
            Threshold to use to remove zero-inflated features from the data
        remove_self_interactions : bool
            Whether to remove self-interactions from the data (i.e. x & y have the same name)
        proximity : np.ndarray
            Proximity matrix to use for the local analysis. If None, will use the one stored in adata.obsp[proximity_key].
        x_use_raw : bool
            Whether to use the raw data for the x modality (By default it will use the .X matrix)
        x_layer : str
            Layer to use for the x modality
        y_use_raw : bool
            Whether to use the raw data for the y modality (By default it will use the .X matrix)
        y_layer : str
            Layer to use for the y modality
        inplace : bool
            Whether to add the results as modalities to to the MuData object
            or return them as a pandas.DataFrame, and local_scores/local_pvalues as a pandas.DataFrame
        Returns
        -------
        
        If inplace is True, it will add the following modalities to the MuData object:
            - local_scores: pandas.DataFrame with the local scores
            - local_pvalues: pandas.DataFrame with the local p-values (if pvalue_method is not None)
            - global_scores: pandas.DataFrame with the global scores
        if inplace is False, it will return:
            - global_scores: pandas.DataFrame with the global scores
            - local_scores: pandas.DataFrame with the local scores
            - local_pvalues: pandas.DataFrame with the local p-values (if pvalue_method is not None)
        
        """
        if pvalue_method not in ['analytical', 'permutation', None]:
            raise ValueError("`pvalue_method` must be one of ['analytical', 'permutation', None]")
        
        xdata = mdata[x_mod]
        xdata.X = _choose_mtx_rep(xdata, use_raw = x_use_raw, layer = x_layer)
        
        ydata = mdata[y_mod]
        ydata.X = _choose_mtx_rep(ydata, use_raw = y_use_raw, layer = y_layer)
        
        proximity = _handle_proximity(mdata, proximity, proximity_key)
        local_fun = _handle_functions(function_name)
        weight = _proximity_to_weight(proximity, local_fun)
        
        # change index names to entity
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
            
        # get local scores
        xy_stats, local_scores, local_pvals = \
            _run_scores_pipeline(xy_stats=xy_stats,
                                 x_mat=x_mat,
                                 y_mat=y_mat,
                                 idx=mdata.obs.index,
                                 local_fun=local_fun,
                                 weight=weight,
                                 seed=seed,
                                 n_perms=n_perms,
                                 pvalue_method=pvalue_method,
                                 positive_only=positive_only,
                                 )
        
        if not inplace:
            return xy_stats, local_scores, local_pvals
            
        # save to uns
        mdata.uns['global_res'] = xy_stats
        
        # save to obsm ## TODO think if this is the best way to do this
        mdata.mod[mod_added] = obsm_to_adata(adata=mdata, df=local_scores, obsm_key=None)
        
        if local_pvals is not None:
            mdata.mod['local_pvals'] = obsm_to_adata(adata=mdata, df=local_pvals, obsm_key=None)

        if add_categories: # TODO move to a pipeline
            # TODO categorizing is currently done following standardization of the matrix
            # i.e. each variable is standardized independently, and then a category is
            # defined based on the values within each stop.
            # (taking into consideration when the input is signed, but not weight or surrounding spots).
            local_catageries = _categorize(_encode_as_char(x_mat.A), _encode_as_char(y_mat.A))
            ## TODO these to helper function that can extract multiple of these
            # and these all saved as arrays, or alternatively saved a modalities in mudata
            mdata.obsm['local_categories'] = _local_to_dataframe(array=local_catageries,
                                                                idx=mdata.obs.index,
                                                                columns=xy_stats.interaction)
            mdata.obsm['local_categories'] = _simplify_cats(mdata.obsm['local_categories'])


basis = SpatialBivariate(_basis_meta)


def _anndata_to_stats(adata, nz_thr=0.1):
    adata.X = csr_matrix(adata.X) ## TODO change to ~prep_check_adata (but not for gene expression alone)
    
    global_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'non_zero': _get_props(adata.X)},
                                index=adata.var_names)
    global_stats = global_stats.reset_index().rename(columns={'index': 'entity'}).rename(columns={'interaction': 'entity'})
    global_stats = global_stats[global_stats['non_zero'] >= nz_thr]

    return global_stats
