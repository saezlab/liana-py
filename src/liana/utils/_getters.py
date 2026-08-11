from __future__ import annotations

import pandas as pd
from anndata import AnnData
from mudata import MuData

from liana._docs import d


@d.dedent
def get_factor_scores(adata: AnnData | MuData,
                      obsm_key: str = None,
                      obs_keys: str | None = None
                      ) -> pd.DataFrame:
    """
    Extract factor scores from an AnnData object.

    Parameters
    ----------
    %(adata)s
    obsm_key
        Key to use when extracting factor scores from `adata.obsm`
    obs_keys
        List of keys to use when extracting metadata from `adata.obs`
        If None, no metadata is extracted. Default is None.

    Returns
    -------
    Returns a pandas DataFrame with the factor scores.

    Raises
    ------
    ValueError
        If `obsm_key` not in `.obsm`

    Examples
    --------
    >>> import scanpy as sc
    >>> from liana.utils._getters import get_factor_scores
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> get_factor_scores(adata, obsm_key='X_pca', obs_keys='bulk_labels')
                    index   Factor1   Factor2  ...  Factor50      bulk_labels
    0    AAAGCCTGGCTAAC-1 -7.939618  5.024914  ...  0.973213   CD14+ Monocyte
    1    AAATTCGATGCACA-1 -7.735241  4.337960  ...  0.071353        Dendritic
    ..                ...       ...       ...  ...       ...              ...
    698  TTCAGTACCGGGAA-8  2.890713 -6.116919  ...  1.076730          CD19+ B
    699  TTGAGGTGGAGAGC-8 -5.787237 -5.075839  ...  0.645973        Dendritic

    [700 rows x 52 columns]

    """
    if obsm_key not in adata.obsm.keys():
        raise ValueError(f'{obsm_key} not found in `.obsm`')

    df = pd.DataFrame(adata.obsm[obsm_key], index=adata.obs.index)

    df.columns = [f'Factor{x + 1}' for x in range(df.shape[1])]
    df = df.reset_index()

    # join with metadata
    if obs_keys is not None:
        obs = adata.obs[obs_keys].reset_index()
        df = df.merge(obs)

    return df

@d.dedent
def get_variable_loadings(adata: AnnData | MuData = None,
                          varm_key:str = None,
                          view_sep:str | None = None,
                          variable_sep:str | None = None,
                          pair_sep:str | None = None,
                          var_names:list = None,
                          pair_names:list = None,
                          drop_columns:bool = True,
                          loadings: pd.DataFrame | dict | None = None,
                          ) -> pd.DataFrame:
    """
    Extract variable loadings from an AnnData object.

    Parameters
    ----------
    %(adata)s
    varm_key
        Key to use when extracting variable loadings from `mdata.varm`.
        Ignored when `loadings` is provided.
    view_sep
        Separator to use when splitting view:variable names into view and
        variable
    variable_sep
        Separator to use when splitting variable names into `var_names`
    pair_sep
        Separator to use when splitting view names into `pair_names`
    var_names
        Variable names given to the splitted variable ('ligand_complex' and
        'receptor_complex' by default)
    pair_names
        Variable names given to the splitted pair ('source' and 'target' by
        default)
    drop_columns
        If True, drop the `view:variable` column
    loadings
        Pre-extracted loadings to use instead of reading from `adata.varm`.
        Either a features-by-factors :class:`~pandas.DataFrame`, or a dict of
        per-view features-by-factors DataFrames (e.g. the output of a MOFA-Flex
        model's ``get_weights()``), which is concatenated feature-wise. When
        provided, `adata` and `varm_key` are ignored and the existing factor
        column names are preserved.

    Returns
    -------
    Returns a pandas DataFrame with the variable loadings for the specified
    index.

    Raises
    ------
    ValueError
        If `varm_key` not found in `.varm` (when `loadings` is not provided)

    Examples
    --------
    >>> import scanpy as sc
    >>> from liana.utils._getters import get_variable_loadings
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> get_variable_loadings(adata, varm_key='PCs')
           index   Factor1  ...  Factor49  Factor50
    440  PTPRCAP  0.136449  ... -0.033913  0.017797
    231     LST1 -0.134028  ...  0.009793  0.009476
    ..       ...       ...  ...       ...       ...
    763    PRMT2       NaN  ...       NaN       NaN
    764   MT-ND3       NaN  ...       NaN       NaN
    """
    if var_names is None:
        var_names = ['ligand_complex', 'receptor_complex']
    if pair_names is None:
        pair_names = ['source', 'target']

    if loadings is not None:
        # loadings supplied directly (e.g. from a MOFA-Flex model's
        # get_weights());
        # a dict of per-view {view: features x factors} is concatenated
        # feature-wise
        if isinstance(loadings, dict):
            loadings = pd.concat(loadings.values(), axis=0)
        df = pd.DataFrame(loadings).copy()
        factor_cols = list(df.columns)
    else:
        if adata is None or varm_key not in adata.varm.keys():
            raise ValueError(f'{varm_key} not found in adata.varm')
        n_factors = adata.varm[varm_key].shape[1]
        factor_cols = [f'Factor{i+1}' for i in range(n_factors)]
        df = pd.DataFrame(
            index=adata.var.index,
            data=adata.varm[varm_key],
            columns=factor_cols
        )

    df.index.name = None
    df = df.reset_index()

    if view_sep:
        df[['view', 'variable']] = df['index'].str.split(view_sep, expand=True)

        if drop_columns:
            df.drop(columns='index', inplace=True)

    if variable_sep:
        if view_sep is None:
            df = df.rename(columns={'index': 'variable'})

        df[var_names] = df['variable'].str.split(variable_sep, expand=True)

        if drop_columns:
            df.drop(columns='variable', inplace=True)

    if pair_sep:
        df[pair_names] = df['view'].str.split(pair_sep, expand=True)

        if drop_columns:
            df.drop(columns='view', inplace=True)

    # Re-order columns so that factors are last
    df = df.reindex(
        sorted(df.columns, key=lambda x: x.startswith('Factor')),
        axis=1
    )

    # re-order to absolute values
    df = (
        df.reindex(df[factor_cols[0]].abs().sort_values(ascending=False).index)
    )

    return df
