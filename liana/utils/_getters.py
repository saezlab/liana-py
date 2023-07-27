import pandas as pd


def get_factor_scores(adata, obsm_key=None, obs_keys=None):
    """
    Extract factor scores from an AnnData object.
    
    Parameters
    ----------
    adata: :class:`~mudata.MuData` or :class:`~anndata.AnnData`
        MuData or Anndata object
    obsm_key: str
        Key to use when extracting factor scores from `adata.obsm`
    obs_keys: list
        List of keys to use when extracting metadata from `adata.obs`
        If None, no metadata is extracted. Default is None.
    
    Returns
    -------
    
    Returns a pandas DataFrame with the factor scores.
    
    """

    if obsm_key not in adata.obsm.keys():
        raise ValueError(f'{obsm_key} not found in mdata.obsm')

    df = pd.DataFrame(adata.obsm[obsm_key], index=adata.obs.index)

    df.columns = ['Factor{0}'.format(x + 1) for x in range(df.shape[1])]
    df = df.reset_index()

    # join with metadata
    if obs_keys is not None:
        obs = adata.obs[obs_keys].reset_index()
        df = df.merge(obs)

    return df


def get_variable_loadings(adata,
                          varm_key = None,
                          view_separator = None,
                          variable_separator = None,
                          pair_separator = None,
                          var_names = ['ligand_complex', 'receptor_complex'],
                          pair_names = ['source', 'target'],
                          drop_columns = True
                          ):
    """
    Extract variable loadings from an AnnData object.
    
    Parameters
    ----------
    
    adata: :class:`~mudata.MuData` or :class:`~anndata.AnnData`
        AnnData or MuData object
    varm_key: str
        Key to use when extracting variable loadings from `mdata.varm`
    view_separator: str
        Separator to use when splitting view:variable names into view and variable
    variable_separator: str
        Separator to use when splitting variable names into `var_names` ('ligand_complex' and 'receptor_complex' by default)
    pair_separator: str
        Separator to use when splitting view names into `pair_names` ('source' and 'target' by default)
    drop_columns: bool
        If True, drop the `view:variable` column
        
    Returns
    -------
    
    Returns a pandas DataFrame with the variable loadings for the specified index.
    """
    if varm_key not in adata.varm.keys():
        raise ValueError(f'{varm_key} not found in adata.varm')
    
    n_factors = adata.varm[varm_key].shape[1]
    columns = [f'Factor{i+1}' for i in range(n_factors)]

    df = pd.DataFrame(index=adata.var.index, data=adata.varm[varm_key], columns=columns)

    df.index.name = None
    df = df.reset_index()

    if view_separator:
        df[['view', 'variable']] = df['index'].str.split(view_separator, expand=True)

        if drop_columns:
            df.drop(columns='index', inplace=True)

    if variable_separator:
        if view_separator is None:
            df = df.rename(columns={'index': 'variable'})
            
        df[var_names] = df['variable'].str.split(variable_separator, expand=True)

        if drop_columns:
            df.drop(columns='variable', inplace=True)

    if pair_separator:
        df[pair_names] = df['view'].str.split(pair_separator, expand=True)

        if drop_columns:
            df.drop(columns='view', inplace=True)

    # Re-order columns so that factors are last
    df = df.reindex(sorted(df.columns, key=lambda x: x.startswith('Factor')), axis=1)

    # re-order to absolute values
    df = (df.reindex(df['Factor1'].abs().sort_values(ascending=False).index))

    return df
