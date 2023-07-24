from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product
from scipy.sparse import csr_matrix
from mudata import MuData
from liana.method._pipe_utils._common import _get_props
from liana.method.sp._SpatialMethod import _SpatialMeta, _basis_meta

from liana.method.sp._spatial_pipe import _categorize, \
    _rename_means, _run_scores_pipeline, \
    _connectivity_to_weight, _handle_connectivity
    
from liana.utils.obsm_to_adata import obsm_to_adata
from liana.utils.mdata_to_anndata import _handle_mdata, mdata_to_anndata
from liana.resource._select_resource import _handle_resource

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
                 resource=None,
                 resource_name=None,
                 remove_self_interactions=True,
                 xy_separator = '^',
                 connectivity_key = 'spatial_connectivities',
                 mod_added = "local_scores",
                 key_added = 'global_res',
                 positive_only=False,
                 add_categories = False,
                 n_perms: int = None,
                 seed = 1337,
                 nz_threshold = 0,
                 x_use_raw = False,
                 x_layer = None,
                 x_transform = False,
                 y_use_raw=False,
                 y_layer = None,
                 y_transform = False,
                 x_name='x_entity',
                 y_name='y_entity',
                 inplace = True,
                 verbose=False,
                 ):
        if n_perms is not None:
            if not isinstance(n_perms, int) or n_perms < 0:
                raise ValueError("n_perms must be None, 0 for analytical or > 0 for permutation")
        
        connectivity = _handle_connectivity(adata=mdata, connectivity_key=connectivity_key)
        local_fun = _handle_functions(function_name)
        weight = _connectivity_to_weight(connectivity, local_fun)
        
        resource = _handle_resource(interactions=interactions,
                                    resource=resource,
                                    resource_name=resource_name,
                                    x_name=x_name, y_name=y_name,
                                    verbose=verbose)
        
        # TODO: change this to mdata_to_anndata
        if isinstance(mdata, MuData):
            xdata, ydata = _handle_mdata(mdata, 
                                         x_mod=x_mod, y_mod=y_mod,
                                         x_use_raw=x_use_raw, x_layer=x_layer,
                                         y_use_raw=y_use_raw, y_layer=y_layer,
                                         x_transform=x_transform, y_transform=y_transform,
                                         verbose=verbose,
                                         )
        
        # TODO: Handle complexes
        
        
        # change index names to entity
        xdata.var_names.rename('entity', inplace=True)
        ydata.var_names.rename('entity', inplace=True)
        
        x_stats = _rename_means(_anndata_to_stats(xdata, nz_threshold), entity='x')
        y_stats = _rename_means(_anndata_to_stats(ydata, nz_threshold), entity='y')
        
        # join global stats to LRs from resource
        xy_stats = resource.merge(x_stats).merge(y_stats)
        
        xy_stats['interaction'] = xy_stats['x_entity'] + xy_separator + xy_stats['y_entity']
        
        # TODO: Should I just get rid of remove_self_interactions?
        self_interactions = xy_stats['x_entity'] == xy_stats['y_entity']
        if self_interactions.any() & remove_self_interactions:
            if verbose:
                print(f"Removing {self_interactions.sum()} self-interactions")
            xy_stats = xy_stats[~self_interactions]
        
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
                                     columns=xy_stats['interaction'],
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
                                 verbose=verbose,
                                 )
        
        if not inplace:
            return xy_stats, local_scores, local_pvals, local_cats
        
        # save to uns
        mdata.uns[key_added] = xy_stats
        # save as a modality
        mdata.mod[mod_added] = obsm_to_adata(adata=mdata, df=local_scores, obsm_key=None, _uns=mdata.uns)
        
        # TODO to a function, passed to .mod if mdata; or obsm if adata
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
