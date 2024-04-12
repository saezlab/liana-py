from __future__ import annotations
from typing import Union, Optional

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
from liana.method.sp._bivariate._global_functions import GlobalFunction
from liana.method.sp._bivariate._local_functions import LocalFunction

from liana._logging import _logg
from liana._docs import d
from liana._constants import Keys as K, DefaultValues as V


class SpatialBivariate():
    """ A class for bivariate local spatial metrics. """
    def __init__(self, x_name: str = 'x', y_name: str = 'y'):
        self.x_name = x_name
        self.y_name = y_name

    @d.dedent
    def __call__(self,
                 mdata: (MuData | AnnData),
                 local_name: (str | None) = 'cosine',
                 global_name: (None | str | list) = None,
                 resource_name: str = None,
                 resource: Optional[pd.DataFrame] = V.resource,
                 interactions: list = V.interactions,
                 connectivity_key: str = K.connectivity_key,
                 mask_negatives: bool = False,
                 add_categories: bool = False,
                 n_perms: int = None,
                 seed: int = V.seed,
                 nz_prop: float = 0.05,
                 remove_self_interactions: bool = True,
                 complex_sep: (None | str) = "_",
                 xy_sep: str = V.lr_sep,
                 inplace: bool = V.inplace,
                 key_added: str = "local_scores",
                 verbose: bool = V.verbose,
                 **kwargs
                 ) -> Union[AnnData, pd.DataFrame] | None:
        """
        A method for bivariate local spatial metrics.

        Parameters
        ----------

        %(mdata)s
        %(local_name)s
        %(global_name)s
        %(interactions)s
        %(resource)s
        %(resource_name)s
        %(connectivity_key)s
        %(mask_negatives)s
        %(add_categories)s
        %(n_perms)s
        %(seed)s
        nz_prop: float
            Minimum proportion of non-zero values for each features. For example, if working with gene expression data,
            this would be the proportion of cells expressing a gene. Both features must have a proportion greater than
            `nz_prop` to be considered in the analysis.
        complex_sep: str
            Separator to use for complex names.
        xy_sep: str
            Separator to use for interaction names.
        remove_self_interactions: bool
            Whether to remove self-interactions. `True` by default.
        %(inplace)s
        key_added: str
            Key in `mdata.mod` (if MuData) or `adata.obsm` (if AnnData) where the local scores will be stored.
        %(verbose)s

        **kwargs : dict, optional
            Additional keyword arguments:
            - For AnnData:
                %(x_name)s By default: 'ligand'.
                %(y_name)s By default: 'receptor'.

            - For MuData:
                %(x_mod)s
                %(y_mod)s
                %(x_name)s By default: 'x'.
                %(y_name)s By default: 'y'.
                x_use_raw: bool
                    Whether to use the raw counts for the x-mod.
                y_use_raw: bool
                    Whether to use the raw counts for y-mod.
                x_layer: str
                    Layer to use for x-mod.
                y_layer: str
                    Layer to use for y-mod.
                x_transform: bool
                    Function to transform the x-mod.
                y_transform: bool
                    Function to transform the y-mod.

        Returns
        -------
        If `inplace` is `True`, the results are added to `mdata` and `None` is returned.
        Note that `var`, `obs`, `obsm`, `uns` and `obsp` attributes are copied to the output object.
        When `mdata` is an `AnnData` object, the results are stored in `mdata.obsm[key_added]`
        When `mdata` is an `MuData` object, the results are stored in `mdata.mod[key_added]`

        `AnnData` objects in `obsm` will not be copied to the output object.

        If `inplace` is `False`, the results are returned.

        """

        if n_perms is not None:
            if not isinstance(n_perms, int) or n_perms < 0:
                raise ValueError("n_perms must be None, 0 for analytical or > 0 for permutation")
        if (n_perms == 0) and ((local_name != "morans") or (global_name=='morans')):
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

        if isinstance(mdata, MuData):
            adata = self._process_mudata(mdata, complex_sep, verbose=verbose, **kwargs)
        elif isinstance(mdata, AnnData):
            adata = self._process_anndata(mdata, complex_sep, verbose=verbose, **kwargs)
        else:
            raise ValueError("Invalid type, `adata/mdata` must be an AnnData/MuData object")

        resource = _handle_resource(interactions=interactions,
                                    resource=resource,
                                    resource_name=resource_name,
                                    x_name=self.x_name,
                                    y_name=self.y_name,
                                    verbose=verbose
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
        xy_stats = xy_stats[(xy_stats[f'{self.x_name}_props'] >= nz_prop) &
                            (xy_stats[f'{self.y_name}_props'] >= nz_prop)]
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
                               uns=adata.uns,
                               obsm=adata.obsm,
                               obsp=adata.obsp,
                               )

        if add_categories:
            local_scores.layers['cats'] = csr_matrix(local_cats)
        if local_pvals is not None:
            local_scores.layers['pvals'] = csr_matrix(local_pvals)

        return self._handle_return(mdata, xy_stats, local_scores, key_added, inplace)

    def _process_anndata(self,
                         adata,
                         complex_sep,
                         verbose,
                         **kwargs):
        expected_params = {'x_name', 'y_name', 'use_raw', 'layer'}
        self.validate_kwargs(expected_params=expected_params, **kwargs)

        self.x_name = kwargs.get('x_name', 'ligand')
        self.y_name = kwargs.get('y_name', 'receptor')

        return prep_check_adata(adata=adata,
                                use_raw=kwargs.get('use_raw', V.use_raw),
                                layer=kwargs.get('layer', V.layer),
                                verbose=verbose,
                                obsm=adata.obsm.copy(),
                                uns=adata.uns.copy(),
                                groupby=None,
                                min_cells=None,
                                complex_sep=complex_sep,
                                )

    def _process_mudata(self,
                        mdata,
                        complex_sep,
                        verbose,
                        **kwargs):
        expected_params = {'x_name', 'y_name',
                           'x_mod', 'y_mod',
                           'x_use_raw', 'x_layer',
                           'y_use_raw', 'y_layer',
                           'x_transform', 'y_transform'}
        self.validate_kwargs(expected_params=expected_params, **kwargs)

        self.x_name = kwargs.get('x_name', self.x_name)
        self.y_name = kwargs.get('y_name', self.y_name)

        x_mod = kwargs.get('x_mod')
        y_mod = kwargs.get('y_mod')

        if x_mod is None or y_mod is None:
            raise ValueError("MuData processing requires 'x_mod' and 'y_mod' parameters.")

        adata = mdata_to_anndata(mdata,
                                 x_mod=x_mod,
                                 y_mod=y_mod,
                                 x_use_raw=kwargs.get('x_use_raw', V.use_raw),
                                 x_layer=kwargs.get('x_layer', V.layer),
                                 y_use_raw=kwargs.get('y_use_raw', V.use_raw),
                                 y_layer=kwargs.get('y_layer', V.layer),
                                 x_transform=kwargs.get('x_transform', False),
                                 y_transform=kwargs.get('y_transform', False),
                                 verbose=verbose
                                 )

        return prep_check_adata(adata=adata,
                                use_raw=False,
                                layer=None,
                                verbose=verbose,
                                obsm = adata.obsm.copy(),
                                uns=adata.uns.copy(),
                                groupby=None,
                                min_cells=None,
                                complex_sep=complex_sep, # NOTE
                                )


    def validate_kwargs(self, expected_params, **kwargs):
        unexpected_kwargs = set(kwargs) - expected_params
        if unexpected_kwargs:
            raise ValueError(f"Unexpected keyword arguments: {unexpected_kwargs}")


    def _rename_means(self, lr_stats, entity):
        df = lr_stats.copy()
        df.columns = df.columns.map(lambda x: entity + '_' + str(x) if x != 'gene' else 'gene')
        return df.rename(columns={'gene': entity})

    def _handle_return(self, data, stats, local_scores, key_added, inplace=False):
        if not inplace:
            return stats, local_scores

        if isinstance(data, MuData):
            data.mod[key_added] = local_scores
        else:
            data.obsm[key_added] = local_scores

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
        cats = x_cats + y_cats
        cats = np.where(cats == 2, 1, np.where(cats == 0, -1, 0))

        return cats

    def show_functions(self):
        """
        Print information about all bivariate local metrics.
        """
        funs = LocalFunction.instances.copy()
        for function in funs.values():
            funs[function.name] = {
                "metadata":function.metadata,
                "reference":function.reference,
                }
        return pd.DataFrame(funs).T.reset_index().rename(columns={"index":"name"})

bivariate = SpatialBivariate()
