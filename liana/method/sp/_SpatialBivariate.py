from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, isspmatrix_csr
from anndata import AnnData
from mudata import MuData

from liana.method._pipe_utils._common import _get_props
from liana.utils.mdata_to_anndata import mdata_to_anndata
from liana.resource.select_resource import _handle_resource
from liana.method._pipe_utils import prep_check_adata, assert_covered

from liana.method.sp._utils import _add_complexes_to_var, _zscore
from liana.method.sp._spatial_pipe import GlobalFunction
from liana.method.sp._bivariate_funs import LocalFunction

from liana._logging import _logg
from liana._docs import d
from liana._constants import Keys as K, DefaultValues as V


class SpatialBivariate():
    """ A class for bivariate local spatial metrics. """
    def __init__(self, x_name='x', y_name='y'):
        self.x_name = x_name
        self.y_name = y_name

    @d.dedent
    def __call__(self,
                 mdata: (MuData | AnnData),
                 x_mod: str,
                 y_mod: str,
                 local_name: (str | None) = 'cosine',
                 global_name: (None | str | list) = None,
                 interactions: (None | list) = None,
                 resource: (None | pd.DataFrame) = None,
                 resource_name: (None | str) = None,
                 connectivity_key: str = K.connectivity_key,
                 mod_added: str = "local_scores",
                 mask_negatives: bool = False,
                 add_categories: bool = False,
                 n_perms: int = None,
                 seed: int = V.seed,
                 nz_threshold: float = 0, # NOTE: do I rename this?
                 x_use_raw: bool = V.use_raw,
                 x_layer: (None | str) = V.layer,
                 x_transform: (bool | callable) = False,
                 y_use_raw: bool = V.use_raw,
                 y_layer: (None | str) = V.layer,
                 y_transform: (bool | callable) = False,
                 complex_sep: (None | str) = None,
                 xy_sep: str = V.lr_sep,
                 remove_self_interactions: bool = True,
                 inplace: bool = V.inplace,
                 verbose: bool = V.verbose,
                 ) -> AnnData | None:
        """
        A method for bivariate local spatial metrics.

        Parameters
        ----------

        %(mdata)s
        %(x_mod)s
        %(y_mod)s
        %(local_name)s
        %(global_name)s
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
        When an MuData object is passed, `obsm`, `varm`, `obsp` and `varp` are copied to `.mod`.
        When `mdata` is an AnnData object, `obsm`, `varm`, `obsp` and `varp` are copied to `.obsm`.
        `AnnData` objects in `obsm` will not be copied to the output object.

        If `inplace` is `False`, the results are returned.

        """

        if n_perms is not None:
            if not isinstance(n_perms, int) or n_perms < 0:
                raise ValueError("n_perms must be None, 0 for analytical or > 0 for permutation")
        if (n_perms == 0) and (local_name != "morans"):
            raise ValueError("An analytical solution is currently available only for Moran's R")
        if global_name is not None:
            if isinstance(global_name, str):
                global_name = [global_name]

            global_funs = GlobalFunction.instances.keys()
            for g_fun in global_funs:
                if g_fun not in global_funs:
                    raise ValueError(f"Invalid global function: {g_fun}. Must be in {global_funs}")

        if local_name is not None:
            local_fun = LocalFunction._get_instance(name=local_name)

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

        weight = self._handle_connectivity(adata=adata, connectivity_key=connectivity_key)

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
        xy_stats = resource.merge(self._rename_means(xy_stats, entity=self.x_name)).merge(
                                    self._rename_means(xy_stats, entity=self.y_name))

        # filter according to props
        xy_stats = xy_stats[(xy_stats[f'{self.x_name}_props'] >= nz_threshold) &
                            (xy_stats[f'{self.y_name}_props'] >= nz_threshold)]
        # create interaction column
        xy_stats['interaction'] = xy_stats[self.x_name] + xy_sep + xy_stats[self.y_name]

        x_mat = adata[:, xy_stats[self.x_name]].X
        y_mat = adata[:, xy_stats[self.y_name]].X

        if global_name is not None:
            for gname in global_name:
                global_fun = GlobalFunction.instances[gname]
                global_fun(xy_stats,
                           x_mat=x_mat,
                           y_mat=y_mat,
                           weight=weight,
                           seed=seed,
                           n_perms=n_perms,
                           mask_negatives=mask_negatives,
                           verbose=verbose,
                           )

        if local_name is None:
            return xy_stats

        # Calculate local scores
        if add_categories or mask_negatives:
            local_cats = self._categorize(x_mat=x_mat,
                                          y_mat=y_mat,
                                          weight=weight,
                                          )
        else:
            local_cats = None

        # get local scores
        local_scores, local_pvals = \
            local_fun(x_mat=x_mat,
                      y_mat=y_mat,
                      weight=weight,
                      seed=seed,
                      n_perms=n_perms,
                      mask_negatives=mask_negatives,
                      verbose=verbose,
                      )

        xy_stats.loc[:, ['mean', 'std']] = \
            np.vstack(
                [np.mean(local_scores, axis=0),
                 np.std(local_scores, axis=0)]
                ).T


        if mask_negatives:
            local_scores = np.where(local_cats!=1, 0, local_scores)
            if local_pvals is not None:
                local_pvals = np.where(local_cats!=1, 1, local_pvals)

        local_scores = AnnData(csr_matrix(local_scores),
                               obs=adata.obs,
                               var=xy_stats.set_index('interaction'),
                               uns=_uns,
                               obsm=adata.obsm,
                               obsp=adata.obsp,
                               )

        if add_categories:
            local_scores.layers['cats'] = csr_matrix(local_cats)
        if local_pvals is not None:
            local_scores.layers['pvals'] = csr_matrix(local_pvals)

        return self._handle_return(mdata, xy_stats, local_scores, mod_added, inplace)

    def _rename_means(self, lr_stats, entity):
        df = lr_stats.copy()
        df.columns = df.columns.map(lambda x: entity + '_' + str(x) if x != 'gene' else 'gene')
        return df.rename(columns={'gene': entity})

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

        if not isspmatrix_csr(connectivity) or (connectivity.dtype != np.float32):
            connectivity = csr_matrix(connectivity, dtype=np.float32)

        return connectivity

    def _encode_cats(self, a, weight):
        if np.all(a >= 0):
            a = _zscore(a)
        a = weight @ a
        a = np.where(a > 0, 1, np.where(a < 0, -1, np.nan))
        return a

    def _categorize(self, x_mat, y_mat, weight):
        x_cats = self._encode_cats(x_mat.A, weight)
        y_cats = self._encode_cats(y_mat.A, weight)

        # add the two categories, and simplify them to ints
        cats = x_cats + y_cats
        cats = np.where(cats == 2, 1, np.where(cats == 0, -1, 0))

        return cats

    def show_functions(self):
        """
        Print information about all bivariate local metrics.
        """
        funs = LocalFunction.instances
        for function in funs.keys():
            funs[function.name] = {
                "metadata":function.metadata,
                "reference":function.reference,
                }

        return pd.DataFrame(funs).T.reset_index().rename(columns={"index":"name"})

bivar = SpatialBivariate()
