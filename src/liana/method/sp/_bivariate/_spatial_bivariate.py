from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix

from liana._constants import DefaultValues as V
from liana._constants import Keys as K
from liana._docs import d
from liana._logging import _logg
from liana.method._pipe_utils import assert_covered
from liana.method._pipe_utils._common import _get_props
from liana.method.sp._bivariate._global_functions import GlobalFunction
from liana.method.sp._bivariate._local_functions import LocalFunction
from liana.method.sp._utils import (
    _add_complexes_to_var,
    _check_instance,
    _handle_connectivity,
    _process_anndata,
    _process_mudata,
    _rename_means,
    _zscore,
)
from liana.resource.select_resource import _handle_resource


class SpatialBivariate:
    """
    A class for bivariate local spatial metrics.

    Parameters
    ----------
    %(x_name)s
    %(y_name)s

    Attributes
    ----------
    %(x_name)s
    %(y_name)s

    """

    @d.dedent
    def __call__(self,
                 mdata: MuData | AnnData,
                 local_name: str | None = 'cosine',
                 global_name: None | str | list[str] = None,
                 resource_name: str = None,
                 resource: pd.DataFrame | None = V.resource,
                 interactions: list[str] = V.interactions,
                 connectivity_key: str = K.connectivity_key,
                 mask_negatives: bool = False,
                 add_categories: bool = False,
                 n_perms: int = None,
                 seed: int = V.seed,
                 nz_prop: float = 0.05,
                 remove_self_interactions: bool = True,
                 complex_sep: None | str = "_",
                 xy_sep: str = V.lr_sep,
                 verbose: bool = V.verbose,
                 **kwargs
                 ) -> AnnData | pd.DataFrame | None:
        """
        A method for bivariate local spatial metrics.

        Parameters
        ----------
        %(mdata)s
        %(local_name)s
        %(global_name)s
        %(resource_name)s
        %(resource)s
        %(interactions)s
        %(connectivity_key)s
        %(mask_negatives)s
        %(add_categories)s
        %(n_perms)s
        %(seed)s
        nz_prop
            Minimum proportion of non-zero values for each features.
            For example, if working with gene expression data,
            this would be the proportion of cells expressing a gene.
            Both features must have a proportion greater than
            `nz_prop` to be considered in the analysis.
        remove_self_interactions
            Whether to remove self-interactions. `True` by default.
        complex_sep
            Separator to use for complex names.
        xy_sep
            Separator to use for interaction names.
        %(verbose)s
        **kwargs
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

        Raises
        ------
        ValueError
            If `n_perms` is not None or negative or if `mdata` is not a valid type.

        Returns
        -------
        An AnnData object, (optionally) with multiple layers which correspond
        categories/p-values, and the actual scores are stored in `.X`.
        Moreover, global stats are stored in ``.var``.

        """
        if n_perms is not None:
            if not isinstance(n_perms, int) or n_perms < 0:
                raise ValueError("n_perms must be None, 0 for analytical or > 0 for permutation")
        if global_name is not None:
            if isinstance(global_name, str):
                global_name = [global_name]
        if (n_perms == 0) and ((local_name not in ["morans", None]) or ~np.isin(global_name, ["morans", None]).any()):
            raise ValueError("An analytical solution is currently available only for Moran's R")

        if local_name is not None:
            local_fun = LocalFunction._get_instance(name=local_name)

        is_mudata = _check_instance(mdata)
        if is_mudata:
            adata, x_name, y_name = _process_mudata(mdata, complex_sep, verbose, **kwargs)
        else:
            adata, x_name, y_name = _process_anndata(mdata, complex_sep, verbose, **kwargs)

        resource = _handle_resource(interactions=interactions,
                                    resource=resource,
                                    resource_name=resource_name,
                                    x_name=x_name,
                                    y_name=y_name,
                                    verbose=verbose
                                    )
        weight = _handle_connectivity(adata=adata, connectivity_key=connectivity_key)

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

        self_interactions = resource[x_name] == resource[y_name]
        if self_interactions.any() & remove_self_interactions:
            _logg(f"Removing {self_interactions.sum()} self-interactions", verbose=verbose)
            resource = resource[~self_interactions]

        # get entities
        entities = np.union1d(np.unique(resource[x_name]),
                                np.unique(resource[y_name]))
        assert_covered(entities, adata.var_names, verbose=verbose)

        # Filter to only include the relevant features
        adata = adata[:, np.intersect1d(entities, adata.var.index)]

        xy_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'props': _get_props(adata.X)},
                                index=adata.var_names
                                ).reset_index().rename(columns={'index': 'gene'})
        # join global stats to LRs from resource
        xy_stats = (
            resource
            .merge(_rename_means(xy_stats, entity=x_name))
            .merge(_rename_means(xy_stats, entity=y_name))
        )

        # filter according to props
        xy_stats = xy_stats[(xy_stats[f'{x_name}_props'] >= nz_prop) &
                            (xy_stats[f'{y_name}_props'] >= nz_prop)]
        if xy_stats.empty:
            raise ValueError("No features with non-zero proportions")

        # create interaction column
        xy_stats['interaction'] = xy_stats[x_name] + xy_sep + xy_stats[y_name]

        x_mat = adata[:, xy_stats[x_name]].X
        y_mat = adata[:, xy_stats[y_name]].X

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

        return local_scores

    def _encode_cats(self, a, weight):
        if np.all(a >= 0):
            a = _zscore(a)
        a = weight @ a
        a = np.where(a > 0, 1, np.where(a < 0, -1, np.nan))
        return a

    def _categorize(self, x_mat, y_mat, weight):
        x_cats = self._encode_cats(x_mat.toarray(), weight)
        y_cats = self._encode_cats(y_mat.toarray(), weight)
        cats = x_cats + y_cats
        cats = np.where(cats == 2, 1, np.where(cats == 0, -1, 0))

        return cats

    def show_functions(self) -> pd.DataFrame:
        """
        Print information about all bivariate local metrics.

        Returns
        -------
        Table of the bivariate methods and their description.

        """
        funs = LocalFunction.instances.copy()
        for function in funs.values():
            funs[function.name] = {
                "metadata":function.metadata,
                "reference":function.reference,
                }
        return pd.DataFrame(funs).T.reset_index().rename(columns={"index":"name"})


bivariate = SpatialBivariate()
