from types import ModuleType

import numpy as np
import pandas as pd
import scanpy as sc

from anndata import AnnData
from tqdm import tqdm

from ..method import get_method_scores


def _check_if_mudata() -> ModuleType:

    try:
        from mudata import MuData

    except Exception:

        raise ImportError(
            'mudata is not installed. Please install it with: '
            'pip install mudata'
        )

    return MuData

## TODO generalize these functions to work with any package
def _check_if_decoupler() -> ModuleType:

    try:
        import decoupler as dc

    except Exception:

        raise ImportError(
            'decoupler is not installed. Please install it with: '
            'pip install decoupler'
        )

    return dc


def adata_to_views(adata, groupby, sample_key, obs_keys, view_separator=':', verbose=False, **kwargs):
    """
    Converts an AnnData object to a MuData object with views that represent an aggregate for each entity in `adata.obs[groupby]`.
    
    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        AnnData object
    groupby:
        Column name in `adata.obs` to group by
    sample_key:
        Column name in `adata.obs` to use as sample key
    obs_keys:
        Column names in `adata.obs` to merge with the MuData object
    view_separator
        Separator to use when assigning `adata.var_names` to views
    verbose
        If True, show progress bar.
    **kwargs
        Keyword arguments used to aggregate the values per cell into views. See `dc.get_pseudobulk` for more details.
    
    Returns
    -------
    Returns a MuData object with views that represent an aggregate for each entity in `adata.obs[groupby]`.
    
    """
    
    # Check if MuData & decoupler are installed
    MuData = _check_if_mudata()
    dc = _check_if_decoupler()
    
    views = adata.obs[groupby].unique()
    views = tqdm(views, disable=not verbose)
    
    padatas = {}
    for view in (views):
        # filter AnnData to view
        temp = adata[adata.obs[groupby] == view]
        # assign view to var_names
        temp.var_names = view + view_separator + temp.var_names
        
        padata = dc.get_pseudobulk(temp,
                                   sample_col=sample_key,
                                   groups_col=None, 
                                   **kwargs
                                   )
        
        # only append views that pass QC
        if 0 not in padata.shape:
            del padata.obs
            padatas[view] = padata
            
    # Convert to MuData
    mdata = MuData(padatas)
    
    # merge metadata
    metadata = adata.obs[[sample_key, *obs_keys]].drop_duplicates()
    mdata.obs = mdata.obs.reset_index(names=sample_key).merge(metadata).set_index(sample_key)
    
    return mdata


def lrs_to_views(adata, 
                 score_key=None, 
                 inverse_fun= lambda x: 1 - x,
                 obs_keys=[],
                 lr_prop=0.5,
                 lr_fill=np.nan,
                 lrs_per_view=20,
                 lrs_per_sample=10,
                 samples_per_view=3,
                 lr_separator='^',
                 cell_separator='&',
                 var_separator=':',
                 uns_key = 'liana_res',
                 sample_key='sample',
                 source_key='source',
                 target_key='target', 
                 ligand_key='ligand_complex',
                 receptor_key='receptor_complex',
                 verbose=False
                 ):
    """
    Converts a LIANA result to a MuData object with views that represent an aggregate for each entity in `adata.obs[groupby]`.
    
    Parameters
    ----------
    adata
        AnnData object with LIANA results in `adata.uns[uns_key]`
    score_key
        Column in `adata.uns[uns_key]` that contains the scores to be used for building the views.
    inverse_fun
        Function that is applied to the scores before building the views. Default is `lambda x: 1 - x` which is used to invert the scores
        reflect probabilities (e.g. magnitude_rank), i.e. such for which lower values reflect higher relevance.
        This is handled automatically for the scores in liana.
    obs_keys
        List of keys in `adata.obs` that should be included in the MuData object. Default is `[]`. 
        These columns should correspond to the number of samples in `adata.obs[sample_key]`.
    lr_prop
        Reflects the minimum required proportion of samples for an interaction to be considered for building the views. Default is `0.5`.
    lr_fill
        Value to fill in for interactions that are not present in a view. Default is `np.nan`.
    lrs_per_sample
        Reflects the minimum required number of interactions in a sample to be considered when building a specific view. Default is `10`.
    lrs_per_view
        Reflects the minimum required number of interactions in a view to be considered for building the views. Default is `20`.
    samples_per_view
        Reflects the minimum required samples to keep a view. Default is `3`.
    lr_separator
        Separator to use for the interaction names in the views. Default is `^`.
    cell_separator
        Separator to use for the cell names in the views. Default is `&`.
    var_separator
        Separator to use for the variable names in the views. Default is `:`.
    uns_key
        Key in `adata.uns` that contains the LIANA results. Default is `'liana_res'`.
    sample_key
        Key in `adata.uns[uns_key]` that contains the sample names. Default is `'sample'`.
    source_key
        Key in `adata.uns[uns_key]` that contains the source names. Default is `'source'`.
    target_key
        Key in `adata.uns[uns_key]` that contains the target names. Default is `'target'`.
    ligand_key
        Key in `adata.uns[uns_key]` that contains the ligand names. Default is `'ligand_complex'`.
    receptor_key
        Key in `adata.uns[uns_key]` that contains the receptor names. Default is `'receptor_complex'`.
    verbose
        If True, show progress bar.
    
    Returns
    -------
    Returns a MuData object with views that represent an aggregate for each entity in `adata.obs[groupby]`.
    
    """
    
    # Check if MuData is installed
    MuData = _check_if_mudata()
    
    if (sample_key not in adata.obs.columns) or (sample_key not in adata.uns[uns_key].columns):
        raise ValueError(f'`{sample_key}` not found in `adata.obs` or `adata.uns[uns_key]`!' +
                         'Please ensure that the sample key is present in both objects.')
    
    if uns_key not in adata.uns_keys():
        raise ValueError(f'`{uns_key}` not found in `adata.uns`! Please run `li.mt.rank_aggregate.by_sample` first.')
    
    liana_res = adata.uns[uns_key].copy()
    
    if (score_key is None) or (score_key not in liana_res.columns):
        raise ValueError(f"Score column `{score_key}` not found in `liana_res`")
    
    if any([key not in adata.obs.keys() for key in obs_keys]):
        raise ValueError(f'`{obs_keys}` not found in `adata.obs`! Please check your input.')
    
    keys = np.array([sample_key, source_key, target_key, ligand_key, receptor_key])
    missing_keys = keys[[ key not in liana_res.columns for key in keys]]
    
    if any(missing_keys):
        raise ValueError(f'`{missing_keys}` not found in `adata.uns[{uns_key}]`! Please check your input.')
    
    # concat columns (needed for MOFA)
    liana_res['interaction'] = liana_res[ligand_key] + lr_separator + liana_res[receptor_key]
    liana_res['ct_pair'] = liana_res[source_key] + cell_separator + liana_res[target_key]
    liana_res = liana_res[[sample_key, 'ct_pair', 'interaction', score_key]]
    
    
    # get scores & invert if necessary
    ## TODO - get rid of redundancy here /w Tensor_c2c function
    scores = get_method_scores()
    if not np.isin(score_key, list(scores.keys())).any():
        raise ValueError(f"Score column {score_key} not found method scores. ")
    
    # reverse if ascending order
    ascending_order = scores[score_key]
    if(ascending_order):
        liana_res[score_key] = inverse_fun(liana_res[score_key])
        
    # count samples per interaction
    count_pairs = (liana_res.
                   drop(columns=score_key).
                   groupby(['interaction', 'ct_pair']).
                   count().
                   rename(columns={sample_key: 'count'}).
                   reset_index()
                   )
    
    sample_n = liana_res[sample_key].nunique()
    
    # Keep only lrs above a certain proportion of samples
    count_pairs = count_pairs[count_pairs['count'] >= sample_n * lr_prop]
    liana_res = liana_res.merge(count_pairs.drop(columns='count') , how='inner')
    
    
    # Keep only samples above a certain number of LRs
    count_lrs = (liana_res.
                 drop(columns=score_key).
                 groupby([sample_key, 'ct_pair']).
                 count().
                 rename(columns={'interaction': 'count'}).
                 reset_index()
                 )
    count_lrs = count_lrs[count_lrs['count'] >= lrs_per_sample]
    liana_res = liana_res.merge(count_lrs.drop(columns='count') , how='inner')
    
    
    # convert to anndata views
    views = liana_res['ct_pair'].unique()
    views = tqdm(views, disable=not verbose)
    
    lr_adatas = {}    
    for view in views:
        lrs_per_ct = liana_res[liana_res['ct_pair']==view]
        lrs_wide = lrs_per_ct.pivot(index='interaction', 
                                    columns=sample_key,
                                    values=score_key)
    
        lrs_wide.index = view + var_separator + lrs_wide.index
        lrs_wide = lrs_wide.replace(np.nan, lr_fill)
        
        if lrs_wide.shape[0] >= lrs_per_view: # check if enough LRs
            temp = _dataframe_to_anndata(lrs_wide)
            
            if (temp.shape[0] >= samples_per_view): # check if enough samples
                lr_adatas[view] = temp
                
    # to mdata
    mdata = MuData(lr_adatas)
    
    # merge metadata, TODO redundancy
    metadata = adata.obs[[sample_key, *obs_keys]].drop_duplicates()
    mdata.obs = mdata.obs.reset_index(names=sample_key).merge(metadata).set_index(sample_key)
    
    return mdata
        

def _dataframe_to_anndata(df):
    obs = pd.DataFrame(index=df.columns)
    var = pd.DataFrame(index=df.index)
    X = np.array(df.values).T
    
    return AnnData(X=X, obs=obs, var=var, dtype=np.float32)



def get_variable_loadings(mdata,
                          idx,
                          varm_key = 'LFs',
                          view_separator = None,
                          variable_separator = None,
                          pair_separator = None,
                          var_names = ['ligand_complex', 'receptor_complex'],
                          pair_names = ['source', 'target'],
                          drop_columns = True
                          ):
    """
    A helper function to extract variable loadings from a MuData object.
    
    Parameters
    ----------
    
    mdata: :class:`~mudata.MuData`
        MuData object
    idx: int
        Index of the variable to extract. Pass index 0 to extract the first Factor.
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
    
    df = sc.get.var_df(mdata, varm_keys=[(varm_key, idx)])
    df = df.reset_index(names='view:variable')
    
    if view_separator is not None:
        df[['view', 'variable']] = df['view:variable'].str.split(view_separator, 1, expand=True)
        
        if drop_columns:
            df.drop(columns='view:variable', inplace=True)
    
    if variable_separator is not None:
        df[var_names] = df['variable'].str.split(variable_separator, 1, expand=True)
        
        if drop_columns:
            df.drop(columns='variable', inplace=True)
        
    
    if pair_separator is not None:
        df[pair_names] = df['view'].str.split(pair_separator, 1, expand=True)
        
        if drop_columns:
            df.drop(columns='view', inplace=True)
    
    df = df.rename(columns={"LFs-{0}".format(0):'loadings'})
    
    # re-order to absolute values
    df = (df.reindex(df['loadings'].abs().sort_values(ascending=False).index))
    
    return df


def get_factor_scores(mdata, obsm_key='X_mofa'):
    """
    A helper function to extract factor scores from a MuData object.
    
    Parameters
    ----------
    mdata: :class:`~mudata.MuData`
        MuData object
    obsm_key: str
        Key to use when extracting factor scores from `mdata.obsm`
        
    Returns
    -------
    Returns a pandas DataFrame with the factor scores.
    
    """
    
    if obsm_key not in mdata.obsm.keys():
        raise ValueError(f'{obsm_key} not found in mdata.obsm')
    
    df = pd.DataFrame(mdata.obsm['X_mofa'], index=mdata.obs.index)
    
    df.columns = ['Factor_{0}'.format(x + 1) for x in range(df.shape[1])]
    df = df.reset_index()
    
    # join with metadata
    df = df.merge(mdata.obs.reset_index())
    
    return df
