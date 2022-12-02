from __future__ import annotations

import pandas as pd
import numpy as np
from anndata import AnnData


def _get_props(X_mask):
    return X_mask.getnnz(axis=0) / X_mask.shape[0]


def obsm_to_adata(adata, obsm_key):
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
    Returns
    -------
    acts : AnnData
        New AnnData object with activities in X.
    """

    obs = adata.obs
    var = pd.DataFrame(index=adata.obsm[obsm_key].columns)
    uns = adata.uns
    obsm = adata.obsm
    X = np.array(adata.obsm[obsm_key], dtype=np.float32)

    return AnnData(X=X, obs=obs, var=var, uns=uns, obsm=obsm)
