from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from liana._docs import d


@d.dedent
def obsm_to_adata(adata: AnnData,
                  obsm_key: str,
                  df: pd.DataFrame | None = None,
                  _uns: dict[str, pd.DataFrame] | None = None,
                  _obsm: dict[str, pd.DataFrame] | None = None,
                  _var: pd.DataFrame | None = None,
                  ) -> AnnData:
    """
    Extracts a dataframe from adata.obsm and returns a new AnnData object with the values stored in X.

    Parameters
    ----------
    %(adata)s
    obsm_key
        `.osbm` key to extract.
    df
        Dataframe with stats per cell/spot. If None, it will be extracted from
        adata.obsm[obsm_key].
    _uns
        Dictionary with uns data. If None, it will be extracted from adata.uns.
    _obsm
        Dictionary with obsm data. If None, it will be extracted from
        adata.obsm.
    _var
        Dataframe with var data. If None, it will be extracted from adata.var.

    Returns
    -------
    An AnnData object with the values stored in X.

    Examples
    --------
    See here `[1]`_ or here `[2]`_.

    .. _[1]: https://liana-py.readthedocs.io/en/latest/notebooks/bivariate.html\
    #extract-cell-type-composition
    .. _[2]: https://liana-py.readthedocs.io/en/latest/notebooks/misty.html#ext\
    ract-cell-type-composition

    """
    if df is None:
        df = pd.DataFrame(adata.obsm[obsm_key])

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
    if _var is None:
        var = pd.DataFrame(index = df.columns)
    else:
        var = _var

    X = np.array(df, dtype=np.float32)

    return AnnData(X=X, obs=obs, var=var, uns=uns, obsm=obsm, obsp=obsp)
