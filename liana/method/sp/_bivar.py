from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product
from scipy.sparse import csr_matrix
from mudata import MuData

from liana.method._pipe_utils._pre import _choose_mtx_rep, _get_props

from liana.method.sp._SpatialMethod import _SpatialMeta, _basis_meta

from liana.method.sp._spatial_pipe import _categorize, \
    _rename_means, _run_scores_pipeline, \
    _connectivity_to_weight, _handle_connectivity
    
from liana.funcomics.obsm_to_adata import obsm_to_adata

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
                 x_mod,
                 y_mod,
                 function_name='cosine',
                 interactions = None,
                 xy_separator = '^',
                 connectivity_key = 'spatial_connectivities', # connectivity_key
                 mod_added = "local_scores",
                 key_added = 'global_res',
                 # TODO: this should not be positive_only
                 # but rather remove low-low??
                 positive_only=False,
                 add_categories = False,
                 n_perms: int = None,
                 seed = 1337,
                 nz_threshold = 0,
                 remove_self_interactions=True,
                 connectivity = None,
                 x_use_raw = False,
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
        interactions : list of tuples
            Interactions to use for the local analysis. If None, all pairwise combinations of all variables in x and y will be used.
            Note that this may be very computationally expensive when working with modalities with many variables.
        connectivity_key : str
            Key to use to retrieve the connectivity matrix from adata.obsp.
        mod_added : str
            Name of the modality to add to the MuData object (in case of inplace=True)
        add_categories : bool
            Whether to add_categories about the local scores or not
        positive_only : bool
            Whether to calculate p-values only for positive correlations. `True` by default.
        n_perms : int
            Number of permutations to use for the p-value calculation (when > 1). If -1, will use the analytical solution.
            If None (default), will not calculate p-values (only local scores).
        seed : int
            Seed to use for the permutation
        nz_threshold : int
            Threshold to use to remove zero-inflated features from the data
        remove_self_interactions : bool
            Whether to remove self-interactions from the data (i.e. x & y have the same name).
            Current metrics implemented here do not make sense for self-interactions.
        connectivity : np.ndarray
            connectivity matrix to use for the local analysis. If None, will use the one stored in adata.obsp[connectivity_key].
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
        If inplace is True, it will add `mod_added` to the MuData object.
            If add_categories is True, `cats` will be added as a layer to the `mod_added` AnnData object.
            If n_perms is not None, `local_pvalues` will be added as a layer to the `mod_added` AnnData object.
        if inplace is False, it will return:
            - global_scores: pandas.DataFrame with the global scores
            - local_scores: pandas.DataFrame with the local scores
            - local_pvalues: pandas.DataFrame with the local p-values (if n_perms is not None)
        
        """
        if n_perms is not None:
            if not isinstance(n_perms, int) or n_perms < 0:
                raise ValueError("n_perms must be None, 0 for analytical or > 0 for permutation")
        
        connectivity = _handle_connectivity(mdata, connectivity, connectivity_key)
        local_fun = _handle_functions(function_name)
        weight = _connectivity_to_weight(connectivity, local_fun)
        
        if isinstance(mdata, MuData):
            xdata = mdata[x_mod]
            xdata.X = _choose_mtx_rep(xdata, use_raw = x_use_raw, layer = x_layer)
            
            ydata = mdata[y_mod]
            ydata.X = _choose_mtx_rep(ydata, use_raw = y_use_raw, layer = y_layer)
        
        if interactions is None:
            interactions = list(product(xdata.var_names, ydata.var_names))
        
        interactions = pd.DataFrame(interactions, columns=['x_entity', 'y_entity'])
        
        # change index names to entity
        xdata.var_names.rename('entity', inplace=True)
        ydata.var_names.rename('entity', inplace=True)
        
        x_stats = _rename_means(_anndata_to_stats(xdata, nz_threshold), entity='x')
        y_stats = _rename_means(_anndata_to_stats(ydata, nz_threshold), entity='y')
        
        
        # join global stats to LRs from resource
        xy_stats = interactions.merge(x_stats).merge(y_stats)
        
        xy_stats['interaction'] = xy_stats['x_entity'] + xy_separator + xy_stats['y_entity']
        
        if remove_self_interactions:
            xy_stats = xy_stats[xy_stats['x_entity'] != xy_stats['y_entity']]
        
        # reorder columns
        xy_stats = xy_stats.reindex(columns=sorted(xy_stats.columns))
        
        # TODO get rid of transpose
        x_mat = mdata[x_mod][:, xy_stats['x_entity']].X.T
        y_mat = mdata[y_mod][:, xy_stats['y_entity']].X.T
            
        if add_categories or positive_only:
            local_cats = _categorize(x_mat=x_mat,
                                     y_mat=y_mat,
                                     weight=weight,
                                     idx=mdata.obs.index,
                                     columns=xy_stats.interaction,
                                     )
            pos_msk = local_cats > 0
        else:
            local_cats = None
            pos_msk = None
        
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
                                 positive_only=positive_only,
                                 pos_msk=pos_msk,
                                 )
        
        if not inplace:
            return xy_stats, local_scores, local_pvals, local_cats
            
        # save to uns
        mdata.uns[key_added] = xy_stats
        # save as a modality
        mdata.mod[mod_added] = obsm_to_adata(adata=mdata, df=local_scores, obsm_key=None, _uns=mdata.uns)
        
        if positive_only:
            mdata.mod[mod_added].X = mdata.mod[mod_added].X * pos_msk.T
        if local_cats is not None:
            mdata.mod[mod_added].layers['cats'] = csr_matrix(local_cats.T)
        if local_pvals is not None: 
            mdata.mod[mod_added].layers['pvals'] = csr_matrix(local_pvals.T)
    


bivar = SpatialBivariate(_basis_meta)


def _anndata_to_stats(adata, nz_thr=0.1):
    adata.X = csr_matrix(adata.X, dtype=np.float32) ## TODO change to ~prep_check_adata (but not for gene expression alone)
    
    global_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'non_zero': _get_props(adata.X)},
                                index=adata.var_names)
    
    global_stats.index.name = None
    global_stats = global_stats.reset_index().rename(columns={'index': 'entity'})
    global_stats = global_stats[global_stats['non_zero'] >= nz_thr]

    return global_stats
