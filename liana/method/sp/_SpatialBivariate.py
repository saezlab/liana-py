from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, isspmatrix_csr
from anndata import AnnData
from mudata import MuData

from liana.method._pipe_utils._common import _get_props
from liana.method.sp._spatial_pipe import (
    _categorize,
    _rename_means,
    _run_scores_pipeline,
    _add_complexes_to_var
    )
from liana.utils.mdata_to_anndata import mdata_to_anndata
from liana.resource.select_resource import _handle_resource
from liana.method._pipe_utils import prep_check_adata, assert_covered
from liana.method.sp._bivariate_funs import _handle_functions, _bivariate_functions

from liana._logging import _logg
from liana._docs import d
from liana._constants import Keys as K, DefaultValues as V


class SpatialBivariate():
    """ A class for bivariate local spatial metrics. """
    def __init__(self, x_name='x', y_name='y'):
        self.x_name = x_name
        self.y_name = y_name

    def _handle_return(self, data, stats, local_scores, x_added, inplace=False):
        if not inplace:
            return stats, local_scores

        if isinstance(data, MuData):
            data.mod[x_added] = local_scores
        else:
            data.obsm[x_added] = local_scores


    def _handle_connectivity(self, adata, connectivity_key):
        if connectivity_key not in adata.obsp.keys():
            raise ValueError(f'No connectivity matrix founds in mdata.obsp[{connectivity_key}]')
        connectivity = adata.obsp[connectivity_key]

        if not isspmatrix_csr(connectivity):
            connectivity = csr_matrix(connectivity, dtype=np.float32)

        return connectivity

    def _connectivity_to_weight(self, connectivity, local_fun):
        if not isspmatrix_csr(connectivity) or (connectivity.dtype != np.float32):
            connectivity = csr_matrix(connectivity, dtype=np.float32)

        if local_fun.__name__ == "_local_morans":
            norm_factor = connectivity.shape[0] / connectivity.sum()
            connectivity = norm_factor * connectivity

        if (connectivity.shape[0] < 5000) | local_fun.__name__.__contains__("masked"):
                return connectivity.A
        else:
            return connectivity


    @d.dedent
    def __call__(self,
                 mdata: (MuData | AnnData),
                 x_mod: str,
                 y_mod: str,
                 function_name: str = 'cosine',
                 interactions: (None | list) = None,
                 resource: (None | pd.DataFrame) = None,
                 resource_name: (None | str) = None,
                 connectivity_key:str = K.connectivity_key,
                 mod_added: str = "local_scores",
                 mask_negatives: bool = False,
                 add_categories: bool = False,
                 n_perms: int = None,
                 seed:int = V.seed,
                 nz_threshold:float = 0, # NOTE: do I rename this?
                 x_use_raw: bool = V.use_raw,
                 x_layer: (None | str) = V.layer,
                 x_transform: (bool | callable) = False,
                 y_use_raw: bool = V.use_raw,
                 y_layer: (None | str) = V.layer,
                 y_transform: (bool | callable) = False,
                 complex_sep: (None | str) = None,
                 xy_sep:str = V.lr_sep,
                 remove_self_interactions: bool = True,
                 inplace:bool = V.inplace,
                 verbose:bool = V.verbose,
                 ):
        """
        A method for bivariate local spatial metrics.

        Parameters
        ----------

        %(mdata)s
        %(x_mod)s
        %(y_mod)s
        %(function_name)s
        %(interactions)s
        %(resource)s
        %(resource_name)s
        %(connectivity_key)s
        mod_added: str
            Key in `mdata.mod` where the local scores are stored.
        %(mask_negatives)s
        %(add_categories)s
        %(n_perms)s
        %(seed)s
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
        complex_sep: str
            Separator to use for complex names.
        xy_sep: str
            Separator to use for interaction names.
        remove_self_interactions: bool
            Whether to remove self-interactions. `True` by default.
        %(inplace)s
        %(verbose)s

        Returns
        -------

        If `inplace` is `True`, the results are added to `mdata` and `None` is returned.
        Note that `obsm`, `varm`, `obsp` and `varp` are copied to the output `AnnData` object.
        When an MuData object is passed, `obsm`, `varm`, `obsp` and `varp` are copied to `mdata.mod`.
        When mdata is an AnnData object, `obsm`, `varm`, `obsp` and `varp` are copied to `mdata.obsm`.
        `AnnData` objects in `obsm` will not be copied to the output object.

        If `inplace` is `False`, the results are returned.

        """

        if n_perms is not None:
            if not isinstance(n_perms, int) or n_perms < 0:
                raise ValueError("n_perms must be None, 0 for analytical or > 0 for permutation")
        if (n_perms == 0) and (function_name != "morans"):
            raise ValueError("An analytical solution is currently available only for Moran's R")

        local_fun = _handle_functions(function_name)

        resource = _handle_resource(interactions=interactions,
                                    resource=resource,
                                    resource_name=resource_name,
                                    x_name=self.x_name,
                                    y_name=self.y_name,
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

        _uns = adata.uns
        adata = prep_check_adata(adata=adata,
                                 use_raw=use_raw,
                                 layer=layer,
                                 verbose=verbose,
                                 obsm = adata.obsm.copy(),
                                 groupby=None,
                                 min_cells=None,
                                 complex_sep=complex_sep,
                                )


        connectivity = self._handle_connectivity(adata=adata, connectivity_key=connectivity_key)
        weight = self._connectivity_to_weight(connectivity=connectivity, local_fun=local_fun)

        if complex_sep is not None:
            adata = _add_complexes_to_var(adata,
                                          np.union1d(resource[self.x_name].astype(str),
                                                     resource[self.y_name].astype(str)
                                                     ),
                                          complex_sep=complex_sep
                                          )

        # filter_resource
        resource = resource[(np.isin(resource[self.x_name], adata.var_names)) &
                            (np.isin(resource[self.y_name], adata.var_names))]

        # NOTE: Should I just get rid of remove_self_interactions?
        self_interactions = resource[self.x_name] == resource[self.y_name]
        if self_interactions.any() & remove_self_interactions:
            _logg(f"Removing {self_interactions.sum()} self-interactions", verbose=verbose)
            resource = resource[~self_interactions]

        # get entities
        entities = np.union1d(np.unique(resource[self.x_name]),
                                np.unique(resource[self.y_name]))
        # Check overlap between resource and adata TODO check if this works
        assert_covered(entities, adata.var_names, verbose=verbose)

        # Filter to only include the relevant features
        adata = adata[:, np.intersect1d(entities, adata.var.index)]

        xy_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'props': _get_props(adata.X)},
                                index=adata.var_names
                                ).reset_index().rename(columns={'index': 'gene'})
        # join global stats to LRs from resource
        xy_stats = resource.merge(_rename_means(xy_stats, entity=self.x_name)).merge(
            _rename_means(xy_stats, entity=self.y_name))

        # TODO: nz_threshold to nz_prop? For consistency with other methods
        # filter according to props
        xy_stats = xy_stats[(xy_stats[f'{self.x_name}_props'] >= nz_threshold) &
                            (xy_stats[f'{self.y_name}_props'] >= nz_threshold)]
        # create interaction column
        xy_stats['interaction'] = xy_stats[self.x_name] + xy_sep + xy_stats[self.y_name]

        x_mat = adata[:, xy_stats[self.x_name]].X.T
        y_mat = adata[:, xy_stats[self.y_name]].X.T

        # reorder columns, NOTE: why?
        xy_stats = xy_stats.reindex(columns=sorted(xy_stats.columns))

        if add_categories or mask_negatives:
            local_cats = _categorize(x_mat=x_mat,
                                     y_mat=y_mat,
                                     weight=weight,
                                     )
        else:
            local_cats = None

        # get local scores
        xy_stats, local_scores, local_pvals = \
            _run_scores_pipeline(xy_stats=xy_stats,
                                 x_mat=x_mat,
                                 y_mat=y_mat,
                                 local_fun=local_fun,
                                 weight=weight,
                                 seed=seed,
                                 n_perms=n_perms,
                                 mask_negatives=mask_negatives,
                                 verbose=verbose,
                                 )

        if mask_negatives:
            local_scores = np.where(local_cats!=1, 0, local_scores)
            if local_pvals is not None:
                local_pvals = np.where(local_cats!=1, 1, local_pvals)

        local_scores = AnnData(csr_matrix(local_scores.T),
                               obs=adata.obs,
                               var=xy_stats.set_index('interaction'),
                               uns=_uns,
                               obsm=adata.obsm,
                               )

        if add_categories:
            local_scores.layers['cats'] = csr_matrix(local_cats.T)
        if local_pvals is not None:
            local_scores.layers['pvals'] = csr_matrix(local_pvals.T)

        return self._handle_return(mdata, xy_stats, local_scores, mod_added, inplace)

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
