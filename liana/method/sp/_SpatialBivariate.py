from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from anndata import AnnData
from mudata import MuData
from liana.method._pipe_utils._common import _get_props

from liana.method.sp._spatial_pipe import _categorize, \
    _rename_means, _run_scores_pipeline, \
    _connectivity_to_weight, _handle_connectivity, _add_complexes_to_var
    
from liana.utils.obsm_to_adata import obsm_to_adata
from liana.utils.mdata_to_anndata import mdata_to_anndata
from liana.resource._select_resource import _handle_resource

from liana.method._pipe_utils import prep_check_adata, assert_covered

from liana.method.sp._bivariate_funs import _handle_functions, _bivariate_functions


class SpatialBivariate():
    """ A class for bivariate local spatial metrics. """
    def __init__(self):
        pass
    def __call__(self,
                 mdata,
                 x_mod,
                 y_mod,
                 function_name='cosine',
                 interactions = None,
                 resource=None,
                 resource_name=None,
                 connectivity_key = 'spatial_connectivities',
                 mod_added = "local_scores",
                 key_added = 'global_res',
                 mask_negatives=False,
                 add_categories = False,
                 n_perms: int = None,
                 seed = 1337,
                 nz_threshold = 0, # NOTE: do I rename this?
                 x_use_raw = False,
                 x_layer = None,
                 x_transform = False,
                 y_use_raw=False,
                 y_layer = None,
                 y_transform = False,
                 # TODO: move some of these to self
                 x_name='x',
                 y_name='y',
                 complex_sep=None,
                 xy_sep = '^',
                 remove_self_interactions=True,
                 inplace = True,
                 verbose=False,
                 ):
        """
        A method for bivariate local spatial metrics.
        
        Parameters
        ----------
        
        mdata: MuData or AnnData
            MuData or AnnData object with spatial coordinates.
        x_mod: str
            Name of the modality to use for the x-axis.
        y_mod: str
            Name of the modality to use for the y-axis.
        function_name: str
            Name of the function to use for the analysis.
        interactions: list
            List of tuples with ligand-receptor pairs `[(ligand, receptor), ...]` to be used for the analysis.
            If passed, it will overrule the resource requested via `resource` and `resource_name`.
        resource: pd.DataFrame
            Resource to use for the analysis. If None, `resource_name` is used.
        resource_name: str
            Name of the resource to use for the analysis. If None, the default resource ('consensus') is used.
        connectivity_key: str
            Key in `mdata.uns` where the spatial connectivities are stored.
        mod_added: str
            Key in `mdata.mod` where the local scores are stored.
        key_added: str
            Key in `mdata.uns` where the global scores are stored.
        mask_negatives: bool
            Whether to mask negative-negative (low-low) or uncategorized interactions.
        add_categories: bool
            Whether to add categories about the local scores
        n_perms: int
            Number of permutations to use for the analysis. If None, no p-values are computed. If 0 analytical p-values are computed.
        seed: int
            Seed for the random number generator.
        nz_threshold: float
            Minimum proportion of cells expressing the ligand and receptor.
        x_use_raw: bool
            Whether to use the raw counts for the x-mod.
        x_layer: str
            Layer to use for x-mod.
        x_transform: bool
            Function to transform the x-mod.
        y_use_raw: bool
            Whether to use the raw counts for y-mod.
        y_layer: str
            Layer to use for y-mod.
        y_transform: bool
            Function to transform the y-mod.
        x_name: str
            Name of the x-mod.
        y_name: str
            Name of the y-mod.
        complex_sep: str
            Separator to use for complex names.
        xy_sep: str
            Separator to use for interaction names.
        remove_self_interactions: bool
            Whether to remove self-interactions. `True` by default.
        inplace: bool
            Whether to add the results to `mdata` or return them.
        verbose: bool
            Verbosity flag.
        
        Returns
        -------
        
        If `inplace` is `True`, the results are added to `mdata` and `None` is returned.
        If `inplace` is `False`, the results are returned.
        """
        
        
        if n_perms is not None:
            if not isinstance(n_perms, int) or n_perms < 0:
                raise ValueError("n_perms must be None, 0 for analytical or > 0 for permutation")
        
        local_fun = _handle_functions(function_name)
        
        resource = _handle_resource(interactions=interactions,
                                    resource=resource,
                                    resource_name=resource_name,
                                    x_name=x_name,
                                    y_name=y_name,
                                    verbose=verbose)
        
        if isinstance(mdata, MuData):
            adata = mdata_to_anndata(mdata,
                                     x_mod=x_mod,
                                     y_mod=y_mod,
                                     x_use_raw=x_use_raw,
                                     x_layer=x_layer,
                                     y_use_raw=y_use_raw,
                                     y_layer=y_layer,
                                     x_transform=x_transform,
                                     y_transform=y_transform,
                                     verbose=verbose,
                                     )
            use_raw = False
            layer = None
        elif isinstance(mdata, AnnData):
            adata = mdata
            use_raw = x_use_raw
            layer = x_layer
        else:
            raise ValueError("Invalid type, `adata/mdata` must be an AnnData/MuData object")
            
        adata = prep_check_adata(adata=adata,
                                 use_raw=use_raw,
                                 layer=layer,
                                 verbose=verbose,
                                 groupby=None,
                                 min_cells=None,
                                 complex_sep=complex_sep,
                                )


        connectivity = _handle_connectivity(adata=adata, connectivity_key=connectivity_key)
        weight = _connectivity_to_weight(connectivity=connectivity, local_fun=local_fun)
        
        if complex_sep is not None:
            adata = _add_complexes_to_var(adata,
                                          np.union1d(resource[x_name].astype(str),
                                                     resource[y_name].astype(str)
                                                     ),
                                          complex_sep=complex_sep
                                          )
        
        # filter_resource
        resource = resource[(np.isin(resource[x_name], adata.var_names)) &
                            (np.isin(resource[y_name], adata.var_names))]
        
        # NOTE: Should I just get rid of remove_self_interactions?
        self_interactions = resource[x_name] == resource[y_name]
        if self_interactions.any() & remove_self_interactions:
            if verbose:
                print(f"Removing {self_interactions.sum()} self-interactions")
            resource = resource[~self_interactions]

        # get entities
        entities = np.union1d(np.unique(resource[x_name]),
                                np.unique(resource[y_name]))
        # Check overlap between resource and adata TODO check if this works
        assert_covered(entities, adata.var_names, verbose=verbose)

        # Filter to only include the relevant features
        adata = adata[:, np.intersect1d(entities, adata.var.index)]
        
        xy_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'props': _get_props(adata.X)},
                                index=adata.var_names
                                ).reset_index().rename(columns={'index': 'gene'})
        # join global stats to LRs from resource
        xy_stats = resource.merge(_rename_means(xy_stats, entity=x_name)).merge(
            _rename_means(xy_stats, entity=y_name))
        
        # TODO: nz_threshold to nz_prop? For consistency with other methods
        # filter according to props
        xy_stats = xy_stats[(xy_stats[f'{x_name}_props'] >= nz_threshold) &
                            (xy_stats[f'{y_name}_props'] >= nz_threshold)]
        # create interaction column
        xy_stats['interaction'] = xy_stats[x_name] + xy_sep + xy_stats[y_name]
        
        x_mat = adata[:, xy_stats[x_name]].X.T
        y_mat = adata[:, xy_stats[y_name]].X.T
        
        # reorder columns, NOTE: why?
        xy_stats = xy_stats.reindex(columns=sorted(xy_stats.columns))
        
        if add_categories or mask_negatives:
            local_cats = _categorize(x_mat=x_mat,
                                     y_mat=y_mat,
                                     weight=weight,
                                     idx=mdata.obs.index,
                                     columns=xy_stats['interaction'],
                                     )
            local_msk = local_cats != 0
        else:
            local_cats = None
            local_msk = None
        
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
                                 mask_negatives=mask_negatives,
                                 local_msk=local_msk,
                                 verbose=verbose,
                                 )
        local_scores = obsm_to_adata(adata=mdata, df=local_scores, obsm_key=None, _uns=mdata.uns)
        local_scores.uns[key_added] = xy_stats
        
        if mask_negatives:
            local_scores.X = local_scores.X * local_msk.T
        if local_cats is not None:
            local_scores.layers['cats'] = csr_matrix(local_cats.T)
        if local_pvals is not None:
            local_scores.layers['pvals'] = csr_matrix(local_pvals.T)

        if not inplace:
            return xy_stats, local_scores
        
        mdata.uns[key_added] = xy_stats
        mdata.mod[mod_added] = local_scores


    def show_functions(self):
        """
        Print information about all bivariate local metrics.
        """
        funs = dict()
        for function in _bivariate_functions:
            funs[function.name] = {
                "metadata":function.metadata,
                "reference":function.reference,
                }
            
        return pd.DataFrame(funs).T.reset_index().rename(columns={"index":"name"})
    
bivar = SpatialBivariate()
