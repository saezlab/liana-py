from __future__ import annotations

import pandas as pd
import numpy as np
from anndata import AnnData


def _get_props(X_mask):
    return X_mask.getnnz(axis=0) / X_mask.shape[0]


def obsm_to_adata(adata, obsm_key, df = None):
    """
    Extracts activities as AnnData object.
    From an AnnData object with source activities stored in `.obsm`, generates a new AnnData object with activities in `X`.
    This allows to reuse many scanpy processing and visualization functions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with activities stored in .obsm.
    obsm_key
        `.osbm` key to extract.
    df : pd.DataFrame
        Dataframe with stats per cell/spot. If None, it will be extracted from adata.obsm[obsm_key].
    Returns
    -------
    acts : AnnData
        New AnnData object with activities in X.
    """

    if df is None:
        df = adata.obsm[obsm_key]
    
    obs = adata.obs
    uns = adata.uns
    obsm = adata.obsm
    obsp = adata.obsp
    
    var = pd.DataFrame(df.columns)
    X = np.array(df, dtype=np.float32)

    return AnnData(X=X, obs=obs, var=var, uns=uns, obsm=obsm, obsp=obsp)
