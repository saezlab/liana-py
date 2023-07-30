from __future__ import annotations

from anndata import AnnData
import pandas as pd
from typing import Optional

from liana.method.sp._SpatialBivariate import SpatialBivariate


class SpatialLR(SpatialBivariate):
    """ A child class of SpatialBivariate for ligand-receptor analysis. """
    def __init__(self):
        super().__init__()

    def __call__(self,
                 adata: AnnData,
                 function_name: str,
                 resource_name: str = 'consensus',
                 resource: Optional[pd.DataFrame] = None,
                 interactions=None,
                 expr_prop: float = 0.05,
                 n_perms: int = None,
                 mask_negatives: bool = False,
                 seed: int = 1337,
                 add_categories: bool = False,
                 use_raw: Optional[bool] = True,
                 layer: Optional[str] = None,
                 connectivity_key = 'spatial_connectivities',
                 inplace=True,
                 key_added='global_res',
                 obsm_added='local_scores',
                 lr_sep='^',
                 verbose: Optional[bool] = False,
                 ):
        """
        Local ligand-receptor interaction metrics and global scores.
        
        Parameters
        ----------
        adata: AnnData
            AnnData object with spatial coordinates.
        function_name: str
            Name of the function to use for the analysis.
        resource_name: str
            Name of the resource to use for the analysis.
        resource: pd.DataFrame
            Resource to use for the analysis. If None, the default resource ('consensus') is used.
        interactions: list
            List of tuples with ligand-receptor pairs `[(ligand, receptor), ...]` to be used for the analysis.
            If passed, it will overrule the resource requested via `resource` and `resource_name`.
        expr_prop: float
            Minimum proportion of cells expressing the ligand and receptor.
        n_perms: int
            Number of permutations to use for the analysis.
        positive_only: bool
            Whether to mask non-positive interactions.
        seed: int
            Seed for the random number generator.
        add_categories: bool
            Whether to add categories about the local scores
        use_raw: bool
            Whether to use .raw attribute of adata. If False, .X is used.
        layer: str
            Name of the layer to use. If None, .X is used.
        connectivity_key: str
            Key in `adata.obsp` where the spatial connectivities are stored.
        inplace: bool
            Whether to add the results to `adata.uns` and `adata.obsm` or return them.
        key_added: str
            Key in `adata.uns` where the global results are stored.
        obsm_added: str
            Key in `adata.obsm` where the local scores are stored.
        lr_sep: str
            Separator to use between ligand and receptor names.
        verbose: bool
            Whether to print progress messages.
        
        Returns
        -------
        Returns `adata` with the following fields, if `inplace=True`:
            - `adata.uns[key_added]` - global results (pd.DataFrame)
            - `adata.obsm[obsm_added]` - local scores (AnnData object)
        if `inplace=False`, returns a the above.
        """
        
        lr_res, local_scores = super().__call__(
            mdata=adata,
            function_name=function_name,
            connectivity_key=connectivity_key,
            resource_name=resource_name,
            resource=resource,
            interactions=interactions,
            nz_threshold=expr_prop,
            n_perms=n_perms,
            mask_negatives=mask_negatives,
            add_categories=add_categories,
            x_mod=True,
            y_mod=True,
            x_use_raw=use_raw,
            x_layer=layer,
            seed=seed,
            verbose=verbose,
            xy_sep=lr_sep,
            x_name='ligand',
            y_name='receptor',
            inplace=False,
            complex_sep='_'
            )
                     
        
        if not inplace:
            return lr_res, local_scores
            
        adata.uns[key_added] = lr_res
        adata.obsm[obsm_added] = local_scores

lr_bivar = SpatialLR()
