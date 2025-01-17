from __future__ import annotations
from typing import Union, Optional, Callable

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

import liana as li
from liana.method._pipe_utils._common import _get_props
from liana.method.sp._utils import _add_complexes_to_var, _rename_means
from liana.method._pipe_utils import assert_covered, prep_check_adata
from liana._logging import _logg
from liana._constants import Keys as K, DefaultValues as V
from liana._docs import d
from liana.resource.select_resource import _handle_resource

class SpatialInflow():
    """
    A class for computing trivariate (source&ligand->receptor) global and spatial spatial metrics.

    Parameters
    ----------
    x_name : str
        Column name in the `resource` DataFrame for ligand genes. Default is 'ligand'.
    y_name : str
        Column name in the `resource` DataFrame for receptor genes. Default is 'receptor'.
    """
    
    def __init__(self, x_name: str = 'ligand', y_name: str = 'receptor'):
        self.x_name = x_name
        self.y_name = y_name
        
    @d.dedent
    def __call__(
        self,
        adata: AnnData,
        groupby: str,
        resource_name: str = None,
        nz_prop: float = 0.001,
        connectivity_key: str = K.connectivity_key,
        resource: Optional[pd.DataFrame] = V.resource,
        interactions: Optional[list] = V.interactions,
        complex_sep: Optional[str, None] = '_',
        transform: Optional[Callable] = None,
        use_raw: Optional[bool, None]=V.use_raw,
        layer: Optional[str, None]=V.layer,
        xy_sep: str = V.lr_sep,
        verbose: bool = V.verbose,
        **kwargs
    ) -> AnnData:
        
        """
        A method for bivariate local spatial metrics.

        Parameters
        ----------

        %(adata)s
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
        transform
            Function used to transform the source-ligand values and receptor values. 
            If None, no transformation is applied.
        %(verbose)s
        **kwargs
            (Optional) Keyword arguments pass to the (optional) tranform function.

        Returns
        -------
        An AnnData object of size (s* l * r *, n), where s correspodns to the cell types passed via the groupby parameter,
        l and r are respectively the ligand and receptors expressed in the data and covered in the resource, and n is the
        number of observations.
        """
        
        # NOTE There are some repetitiions with bivariate scores
        # one could define a shared class to process adata, and split the two thereafter
        resource = _handle_resource(interactions=interactions,
                                    resource=resource,
                                    resource_name=resource_name,
                                    x_name=self.x_name,
                                    y_name=self.y_name,
                                    verbose=verbose
                                    )
        
        adata = prep_check_adata(adata=adata,
                                use_raw=use_raw,
                                layer=layer,
                                verbose=verbose,
                                obsm=adata.obsm.copy(),
                                uns=adata.uns.copy(),
                                groupby=None,
                                min_cells=None,
                                complex_sep=complex_sep,
                                )
        
        if complex_sep is not None:
            adata = _add_complexes_to_var(
                adata,
                np.union1d(
                    resource[self.x_name].astype(str),
                    resource[self.y_name].astype(str)
                ),
                complex_sep=complex_sep
            )

        # Filter the resource to keep only rows where both ligand & receptor are in adata.var_names
        resource = resource[
            (resource[self.x_name].isin(adata.var_names)) &
            (resource[self.y_name].isin(adata.var_names))
        ]

        # Make sure all LR features appear in adata.var
        entities = np.union1d(resource[self.x_name].unique(), resource[self.y_name].unique())
        assert_covered(entities, adata.var_names, verbose=verbose)

        # Subset adata to only the relevant (ligand + receptor) features
        adata = adata[:, np.intersect1d(entities, adata.var_names)]

        # Build cell-type matrix
        celltypes = pd.get_dummies(adata.obs[groupby])
        ct = csr_matrix(celltypes.astype(int).values)

        # Compute global stats (proportions) for all features in adata
        xy_stats = pd.DataFrame(
            {
                'props': _get_props(adata.X)
            },
            index=adata.var_names
        ).reset_index().rename(columns={'index': 'gene'})

        # Merge these stats into the resource
        xy_stats = resource.merge(_rename_means(xy_stats, entity=self.x_name)) \
                           .merge(_rename_means(xy_stats, entity=self.y_name))

        # Filter by non-zero proportion
        xy_stats = xy_stats[
            (xy_stats[f'{self.x_name}_props'] >= nz_prop) &
            (xy_stats[f'{self.y_name}_props'] >= nz_prop)
        ]
        if xy_stats.empty:
            raise ValueError("No features passed the non-zero proportion filter.")

        # Create 'interaction' column
        xy_stats['interaction'] = (
            xy_stats[self.x_name] + xy_sep + xy_stats[self.y_name]
        )

        # Extract ligand and receptor expression data
        x_mat = adata[:, xy_stats[self.x_name]].X
        y_mat = adata[:, xy_stats[self.y_name]].X

        # Grab the spatial connectivity matrix
        if 'spatial_connectivities' not in adata.obsp:
            raise ValueError("`adata.obsp` must contain 'spatial_connectivities' for weighting.")
        w = adata.obsp['spatial_connectivities']

        k = ct.shape[1]           # number of cell types
        m = x_mat.shape[1]       # number of LR pairs

        # Expand ligand expression by cell types
        l = np.expand_dims(x_mat.toarray(), axis=1)     # shape: (n_cells, 1, n_LRs)
        s = np.expand_dims(ct.toarray(), axis=2)        # shape: (n_cells, n_ct, 1)
        ls = np.multiply(l, s).reshape((x_mat.shape[0], m * k))

        # Min-max transform the ligand * celltype data & apply spatial weighting
        ls = self._transform(ls, transform, **kwargs)
        wls = w.dot(csr_matrix(ls)).toarray()
        wls /= w.sum(axis=1)  # normalize by row sum
        wls[np.isnan(wls)] = 0
        r = self._transform(y_mat, transform, **kwargs)
        ri = np.tile(r, k)
        xy_mat = wls * ri

        # Create .var index: each column is "cell_type ^ interaction_name"
        var = pd.DataFrame(
            index=(
                np.repeat(celltypes.columns.astype(str), m) +
                xy_sep +
                np.tile(xy_stats['interaction'].astype(str), k)
            )
        )

        # Construct the output AnnData
        lrdata = sc.AnnData(
            X=xy_mat,
            var=var,
            obs=adata.obs,
            uns=adata.uns,
            obsm=adata.obsm,
            varm=adata.varm,
            obsp=adata.obsp
        )

        # Drop non-variable features
        lrdata = lrdata[:, lrdata.X.var(axis=0) > 0]
        
        ## Global Values
        celltype_names = list(celltypes.columns.values) # k
        interaction_names = lrdata.var.index # s x l x r

        t = ct.toarray() # k, n
        t = t / t.sum(axis=0) # NOTE: scaled by the sum of each cell type

        values = (t.T @ lrdata.X) # k x (s*l*r); i.e. we sum over n
        values = values.flatten()

        res = pd.DataFrame({
            'name': np.tile(interaction_names, k)+ xy_sep + np.repeat(celltype_names, interaction_names.shape[0]),
            'value': values
        })

        return res, lrdata
    
    def _transform(self, mat, transform=None, **kwargs):
        if transform is not None:
            mat = transform(mat, **kwargs)
        if isinstance(mat, csr_matrix):
            return mat.toarray()
        elif isinstance(mat, np.ndarray):
            return mat
        else:
            raise TypeError(f"Unsupported matrix type: {type(mat)}. Expected np.ndarray or csr_matrix.")

inflow = SpatialInflow()