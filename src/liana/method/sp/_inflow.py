from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix, hstack
from sklearn.utils.sparsefuncs import mean_variance_axis

from liana._constants import DefaultValues as V
from liana._constants import Keys as K
from liana._docs import d
from liana.method._pipe_utils import assert_covered
from liana.method._pipe_utils._common import _get_props
from liana.method.sp._utils import (
    _add_complexes_to_var,
    _check_instance,
    _handle_connectivity,
    _process_anndata,
    _process_mudata,
    _rename_means,
)
from liana.resource.select_resource import _handle_resource


class SpatialInflow:
    """A class for computing trivariate (source&ligand->receptor) global and spatial spatial metrics."""

    @d.dedent
    def __call__(
        self,
        adata: AnnData | MuData,
        groupby: str | None = None,
        obsm_key: str | None = None,
        resource_name: str = None,
        resource: pd.DataFrame | None = V.resource,
        interactions: list = V.interactions,
        nz_prop: float = 0.001,
        connectivity_key: str = K.connectivity_key,
        complex_sep: str | None = V.complex_sep,
        x_transform: Callable | None = None,
        y_transform: Callable | None = None,
        use_raw: bool | None=V.use_raw,
        layer: str | None=V.layer,
        xy_sep: str = V.lr_sep,
        verbose: bool = V.verbose,
        **kwargs
    ) -> AnnData:
        """
        A method for trivariate (source cell type, ligand, receptor) local spatial metrics.

        Parameters
        ----------
        %(adata)s
        groupby : str, optional
            Column name in `adata.obs` containing cell type labels. If provided, a one-hot encoding
            will be created. Mutually exclusive with `obsm_key`.
        obsm_key : str, optional
            Key in `adata.obsm` containing a pre-computed cell type matrix (pandas DataFrame) of shape
            (n_obs, n_celltypes). Column names will be used as cell type labels. Can contain binary
            (one-hot) or continuous (probabilities/scores) values. Mutually exclusive with `groupby`.
        %(interactions)s
        %(resource)s
        %(resource_name)s
        %(connectivity_key)s
        %(mask_negatives)s
        %(add_categories)s
        %(layer)s
        %(use_raw)s
        nz_prop: float
            Minimum proportion of non-zero values for each features. For example, if working with gene expression data,
            this would be the proportion of cells expressing a gene. Both features must have a proportion greater than
            `nz_prop` to be considered in the analysis.
        complex_sep: str
            Separator to use for complex names.
        xy_sep: str
            Separator to use for interaction names.
        x_transform
            Function used to transform the source-ligand values.
            If None, no transformation is applied.
        y_transform
            Function used to transform the receptor values.
            If None, no transformation is applied.
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
                    Whether to use raw counts for x modality.
                y_use_raw: bool
                    Whether to use raw counts for y modality.
                x_layer: str
                    Layer to use for x modality.
                y_layer: str
                    Layer to use for y modality.

            - For both AnnData and MuData:
                x_transform_kwargs: dict
                    Keyword arguments to pass to x_transform function.
                y_transform_kwargs: dict
                    Keyword arguments to pass to y_transform function.

        Returns
        -------
        An AnnData object of shape (n_cell_type_ligand_receptor_combinations, n_observations), where
        n_cell_type_ligand_receptor_combinations corresponds to the combinations of cell types (as defined by the
        ``groupby`` parameter) with ligands and receptors expressed in the data and covered by the resource, and
        n_observations is the number of observations.
        """
        # Process MuData or AnnData - check instance and process accordingly
        is_mudata = _check_instance(adata)

        # Extract transform kwargs - works the same for both AnnData and MuData
        x_transform_kwargs = kwargs.pop('x_transform_kwargs', {})
        y_transform_kwargs = kwargs.pop('y_transform_kwargs', {})

        if is_mudata:
            # For MuData: convert to AnnData, transformations after l * s, not in _process_mudata
            kwargs.setdefault('x_transform', None)
            kwargs.setdefault('y_transform', None)
            adata, x_name, y_name = _process_mudata(
                adata, complex_sep, verbose,
                # NOTE: transformation after l * s
                **kwargs
            )
        else:
            # For AnnData: standard processing
            adata, x_name, y_name = _process_anndata(
                adata, complex_sep, verbose,
                use_raw=use_raw,
                layer=layer,
                **kwargs
            )

        # NOTE: There are some repetitions with bivariate scores
        # one could define a shared class to process adata, and split the two thereafter
        resource = _handle_resource(interactions=interactions,
                                    resource=resource,
                                    resource_name=resource_name,
                                    x_name=x_name,
                                    y_name=y_name,
                                    verbose=verbose
                                    )

        if complex_sep is not None:
            adata = _add_complexes_to_var(
                adata,
                np.union1d(
                    resource[x_name].astype(str),
                    resource[y_name].astype(str)
                ),
                complex_sep=complex_sep
            )

        # Filter the resource to keep only rows where both ligand & receptor are in adata.var_names
        resource = resource[
            (resource[x_name].isin(adata.var_names)) &
            (resource[y_name].isin(adata.var_names))
        ]

        # Make sure all LR features appear in adata.var
        entities = np.union1d(resource[x_name].unique(), resource[y_name].unique())
        assert_covered(entities, adata.var_names, verbose=verbose)

        # Subset adata to only the relevant (ligand + receptor) features
        adata = adata[:, np.intersect1d(entities, adata.var_names)]

        # Validate that exactly one of groupby or obsm_key is provided
        if (groupby is None) == (obsm_key is None):
            raise ValueError("Exactly one of 'groupby' or 'obsm_key' must be provided.")

        # Build cell-type matrix
        if obsm_key is not None:
            # Use pre-computed cell type probabilities from obsm
            if obsm_key not in adata.obsm:
                raise KeyError(f"'{obsm_key}' not found in adata.obsm")

            ct_matrix = adata.obsm[obsm_key]
            if not isinstance(ct_matrix, pd.DataFrame):
                raise TypeError(f"obsm['{obsm_key}'] must be a pandas DataFrame with cell type labels as column names")

            ct_labels = ct_matrix.columns
            ct = csr_matrix(ct_matrix.values)
        else:
            # Existing one-hot encoding logic
            celltypes = pd.get_dummies(adata.obs[groupby])
            ct_labels = celltypes.columns
            ct = csr_matrix(celltypes.astype(int).values)

        # Compute global stats (proportions) for all features in adata
        xy_stats = pd.DataFrame(
            {
                'props': _get_props(adata.X)
            },
            index=adata.var_names
        ).reset_index().rename(columns={'index': 'gene'})

        xy_stats.rename(columns={xy_stats.columns[0]: 'gene'}, inplace=True)

        # Merge these stats into the resource
        # NOTE: add to .var?
        xy_stats = resource.merge(_rename_means(xy_stats, entity=x_name)) \
                           .merge(_rename_means(xy_stats, entity=y_name))

        # Filter by non-zero proportion
        xy_stats = xy_stats[
            (xy_stats[f'{x_name}_props'] >= nz_prop) &
            (xy_stats[f'{y_name}_props'] >= nz_prop)
        ]
        if xy_stats.empty:
            raise ValueError("No features passed the non-zero proportion filter.")

        # Create 'interaction' column
        xy_stats['interaction'] = (
            xy_stats[x_name] + xy_sep + xy_stats[y_name]
        )

        # Extract ligand and receptor expression data
        x_mat = adata[:, xy_stats[x_name]].X
        y_mat = adata[:, xy_stats[y_name]].X

        # Grab the spatial connectivity matrix using utility function
        w = _handle_connectivity(adata=adata, connectivity_key=connectivity_key)

        k = ct.shape[1]           # number of cell types
        m = x_mat.shape[1]       # number of LR pairs

        # Initialize empty list to hold each (cell x ligand) matrix per celltype
        ls_list = []

        # Loop over each cell type column in `ct` (a sparse binary matrix)
        for i in range(ct.shape[1]):
            # Slice the indicator column for one cell type: (n_cells, 1)
            ct_i = ct[:, i]

            # Elementwise multiply x_mat (ligand expr) with cell type indicator
            # This will zero out cells not in this cell type
            ls_i = x_mat.multiply(ct_i)

            ls_list.append(ls_i)

        # Horizontally stack to simulate (n_cells, n_celltypes * n_ligands)
        ls = hstack(ls_list)  # shape: (n_cells, k * m)

        # Min-max transform the ligand * celltype data & apply spatial weighting
        if not isinstance(ls, csr_matrix):
            ls = csr_matrix(ls)

        # Transform ligand matrix
        ls = self._transform(ls, x_transform, **x_transform_kwargs)

        wls = w.dot(ls)

        # Normalize by row sums (avoid division by zero)
        row_sums = np.asarray(w.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        inv_row_sums = 1.0 / row_sums
        wls = wls.multiply(inv_row_sums.reshape(-1, 1))  # still sparse

        # Clean NaNs in sparse matrix (if any)
        wls.data[np.isnan(wls.data)] = 0

        # Transform receptor matrix
        r = self._transform(y_mat, y_transform, **y_transform_kwargs)

        # Ensure r is sparse and repeat across cell types
        if not isinstance(r, csr_matrix):
            r = csr_matrix(r)
        ri = hstack([r] * k)  # replicate across k cell types - changed

        # Sparse elementwise multiplication
        xy_mat = wls.multiply(ri)  # both are sparse


        # Create .var index: each column is "cell_type ^ interaction_name"
        var = pd.DataFrame(
            index=(
                np.repeat(ct_labels.astype(str), m) +
                xy_sep +
                np.tile(xy_stats['interaction'].astype(str), k)
            )
        )

        # Construct the output AnnData
        lrdata = sc.AnnData(
            X=csr_matrix(xy_mat),
            var=var,
            obs=adata.obs,
            uns=adata.uns,
            obsm=adata.obsm,
            varm=adata.varm,
            obsp=adata.obsp
        )

        # Drop non-variable features
        _, var = mean_variance_axis(lrdata.X, axis=0)
        lrdata = lrdata[:, var > 0]

        X = lrdata.X.astype(float)
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
        var = mean_sq - mean**2
        std = np.sqrt(var)
        cv = std / (mean + 1e-12)

        lrdata.var["mean"] = mean
        lrdata.var["variance"] = var
        lrdata.var["std"] = std
        lrdata.var["cv"] = cv
        lrdata.var["nonzero_fraction"] = (lrdata.X.astype(bool).sum(axis=0) / lrdata.n_obs).A1


        return lrdata

    def _transform(self, mat, transform=None, **kwargs):
        if transform is not None:
            return transform(mat, **kwargs)
        return mat


inflow = SpatialInflow()
