import numpy as np
import pandas as pd
from anndata import AnnData

def obsm_to_adata(adata, obsm_key, df = None, _uns=None, _obsm=None):
    """
    Extracts activities as AnnData object.
    From an AnnData object with source activities stored in `.obsm`, generates a new AnnData object with activities in `X`.
    This allows to re-use many scanpy processing and visualization functions.
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with activities stored in .obsm.
    obsm_key
        `.osbm` key to extract.
    df : pd.DataFrame
        Dataframe with stats per cell/spot. If None, it will be extracted from adata.obsm[obsm_key].
    _uns : AxisArrays
        Dictionary with uns data. If None, it will be extracted from adata.uns.
    _obsm : AxisArrays
        Dictionary with obsm data. If None, it will be extracted from adata.obsm.
    Returns
    -------
    acts : AnnData
        New AnnData object with activities in X.
    """

    if df is None:
        df = adata.obsm[obsm_key]
    
    obs = adata.obs
    
    if _uns is None:
        uns = adata.uns
    else:
        uns = _uns
        
    if _obsm is None:
        obsm = adata.obsm
    else:
        obsm = _obsm
    
    obsp = adata.obsp
    
    var = pd.DataFrame(index = df.columns)
    X = np.array(df, dtype=np.float32)

    return AnnData(X=X, obs=obs, var=var, uns=uns, obsm=obsm, obsp=obsp)