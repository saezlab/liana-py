from __future__ import annotations

from anndata import AnnData
import pandas as pd
from typing import Optional

from liana.method.sp._bivar import SpatialBivariate


class SpatialLR(SpatialBivariate):
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
                 positive_only: bool = False,
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
        
        lr_res, local_scores = super().__call__(
            mdata=adata,
            function_name=function_name,
            connectivity_key=connectivity_key,
            resource_name=resource_name,
            resource=resource,
            interactions=interactions,
            nz_threshold=expr_prop,
            n_perms=n_perms,
            positive_only=positive_only,
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
            inplace=False
            )
                     
        
        if not inplace:
            return lr_res, local_scores
            
        adata.uns[key_added] = lr_res
        adata.obsm[obsm_added] = local_scores

lr_bivar = SpatialLR()
