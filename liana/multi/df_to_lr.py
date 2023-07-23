import numpy as np
import pandas as pd
from itertools import product

from liana.method._pipe_utils import prep_check_adata, assert_covered, filter_resource, \
    filter_reassemble_complexes
from liana.method._pipe_utils._common import _join_stats
from liana.method._pipe_utils._common import _get_props
from liana.resource import _handle_resource, explode_complexes

def df_to_lr(adata,
              dea_df,
              groupby,
              stat_keys,
              resource_name = 'consensus',
              resource = None,
              interactions = None,
              layer = None, 
              use_raw = True,
              expr_prop = 0.1,
              min_cells=10,
              complex_col=None,
              return_all_lrs=False,
              source_labels = None,
              target_labels = None,
              lr_sep="^",
              verbose = False,
              ):
    """
    Convert DEA results to ligand-receptor pairs.
    
    Parameters
    ----------
    adata : AnnData object
        Anndata Object
    dea_df : pd.DataFrame
        DEA results. Index must match adata.var_names
    groupby : str
        Column in adata.obs and dea_df to groupby
    stat_keys : list
        List of statistics to use for lr pairs
    resource_name : str, optional
        Name of resource to use. Default is 'consensus'
    resource : pd.DataFrame, optional
        Resource to use. If None, will use resource_name
    interactions : pd.DataFrame, optional
        Interactions to use. If None, will use resource
    layer : str, optional
        Layer to use. If None, will use adata.raw.X
    use_raw : bool, optional
        Whether to use adata.raw.X. Default is True
    expr_prop : float, optional
        Minimum cells expressing a gene for it to be considered expressed. Default is 0.1
    min_cells : int, optional
        Minimum cells per group to be considered. Default is 10
    complex_col : str, optional
        Column in dea_df to use for complex expression. Default is None.
        If None, will use mean expression ('expr') calculated per group in `groupby`.
    return_all_lrs : bool, optional
        Whether to return all ligand-receptor pairs. Default is False.
        If False, will only return ligand-receptor pairs, the genes for which
        have matching statistics in dea_df (for each group in `groupby`).
    source_labels : list, optional
        List of labels to use as source. Default is None
    target_labels : list, optional
        List of labels to use as target. Default is None
    lr_sep : str, optional
        Separator to use between ligand and receptor. Default is '^'
    verbose : bool, optional
        Whether to print progress. Default is False
    
    Returns
    -------
    Returns a pd.DataFrame with joined ligand-receptor pairs and statistics.        

    """
    
    _key_cols = ['source', 'target', 'ligand_complex', 'receptor_complex']
        
    if (groupby not in adata.obs.columns) or (groupby not in dea_df.columns):
        raise ValueError(f'groupby must match a column in adata.obs and dea_df')
    if not np.any(adata.var_names.isin(dea_df.index)):
        raise ValueError(f'index of dea_df must match adata.var_names')
    
    resource = _handle_resource(interactions=interactions,
                                resource=resource,
                                resource_name=resource_name,
                                verbose=verbose)
    
    stat_names = ['expr', 'props'] + stat_keys
    if complex_col is not None:
        if complex_col not in stat_names:
            raise ValueError(f'complex_col must be one of {stat_names}')
        stat_names = stat_names[stat_names.index(complex_col):]+stat_names[:stat_names.index(complex_col)]
    else:
        complex_col = 'expr'
    
    # Check and Reformat Mat if needed
    adata = prep_check_adata(adata=adata,
                             groupby=groupby,
                             min_cells=min_cells,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose
                             )
    
    # reduce dim of adata
    adata =  adata[:, dea_df.index.unique()]
    
    # get label cats
    labels = adata.obs['label'].cat.categories
    
    dedict = {}
    
    for label in labels:
        temp = adata[adata.obs.label == label, :]
        props = _get_props(temp.X)
        means = np.array(temp.X.mean(axis=0), dtype='float32').flatten()
        
        stats = pd.DataFrame({'names': temp.var_names,
                              'props': props,
                              'expr': means
                              }). \
            assign(label=label).sort_values('names')
            
        # merge DEA results to props & means
        dea_df.index.name = None
        df = dea_df[dea_df[groupby] == label].drop(groupby, axis=1). \
            reset_index().rename(columns={'index': "names"})
        
        if not return_all_lrs:
            stats = stats.merge(df, on='names')
        else: 
            stats = df.merge(stats, on='names', how='outer')
            
        dedict[label] = stats[['names', 'label', *stat_names]]
        all_stats = pd.concat(dedict.values())
        
    # Create df /w cell identity pairs
    pairs = pd.DataFrame(list(product(labels, labels))). \
        rename(columns={0: "source", 1: "target"})
    
    if source_labels is not None:
        pairs = pairs[pairs['source'].isin(source_labels)]
    if target_labels is not None:
        pairs = pairs[pairs['target'].isin(target_labels)]
        
    resource = explode_complexes(resource)
    
    # Check overlap between resource and adata
    assert_covered(np.union1d(np.unique(resource["ligand"]),
                                np.unique(resource["receptor"])),
                    all_stats['names'], verbose=verbose)

    # Filter Resource
    resource = filter_resource(resource, all_stats['names'].unique())
    
    # Join Stats to LR
    lr_res = pd.concat(
        [_join_stats(source, target, dedict, resource) for
            source, target in zip(pairs['source'], pairs['target'])]
    )
    
    # ligand_ or receptor + stat_keys
    complex_cols = list(product(['ligand', 'receptor'], [complex_col]))
    complex_cols = [f'{x}_{y}' for x, y in complex_cols]
    
    # assign receptor and ligand absolutes, NOTE deals with missing values
    _placeholders = ['ligand_absolute', 'receptor_absolute']
    lr_res[_placeholders] = \
        lr_res[complex_cols].apply(lambda x: x.abs())
    if return_all_lrs:
        lr_res[_placeholders] = lr_res[_placeholders].fillna(0)
    
    lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                         _key_cols=_key_cols,
                                         expr_prop=expr_prop,
                                         return_all_lrs=return_all_lrs,
                                         complex_cols=_placeholders
                                        )
    lr_res = lr_res.drop(['prop_min', 'interaction', *_placeholders], axis=1)
    
    # summarise stats for each lr
    for key in stat_keys:
        stat_columns = lr_res.columns[lr_res.columns.str.endswith(key)]
        lr_res.loc[:, f'interaction_{key}'] = lr_res.loc[:, stat_columns].mean(axis=1)
        
    lr_res['interaction'] = lr_res['ligand_complex'] + lr_sep + lr_res['receptor_complex']
    
    return lr_res
