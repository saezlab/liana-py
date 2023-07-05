import numpy as np
import pandas as pd
from itertools import product

from liana.method._pipe_utils import prep_check_adata, assert_covered, filter_resource, \
    filter_reassemble_complexes
from liana.resource import select_resource, explode_complexes
from liana.method._liana_pipe import _get_props, _join_stats


def dea_to_lr(adata,
              dea_df,
              groupby,
              stat_keys,
              resource_name = 'consensus',
              resource = None,
              layer = None, 
              use_raw = True,
              expr_prop = 0.1,
              min_cells=10,
              return_all_lrs=False,
              source_labels = None,
              target_labels = None,
              verbose = False,
              ):
    
    if resource is None:
        if resource_name is not None:
            resource = select_resource(resource_name)
    else:
        raise ValueError('Please provide a `resource` or a valid `resource_name`')
    
    stat_names = stat_keys + ['props', 'expr']
    _key_cols = ['source', 'target', 'ligand_complex', 'receptor_complex']
    
    # Check and Reformat Mat if needed
    adata = prep_check_adata(adata=adata,
                             groupby=groupby,
                             min_cells=min_cells,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose
                             )
    
    # reduce dim of adata
    adata =  adata[:, dea_df.index.unique()].copy()
    
    # get label cats
    labels = adata.obs.label.cat.categories
    
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
        df = dea_df[dea_df[groupby] == label].drop(groupby, axis=1). \
            reset_index().rename(columns={'index': "names"})
            
        stats = stats.merge(df, on='names')
        dedict[label] = stats[['names', 'label', *stat_names]]
        
        
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
                    dea_df.index, verbose=verbose)

    # Filter Resource
    resource = filter_resource(resource, dea_df.index)
    
    # Join Stats to LR
    lr_res = pd.concat(
        [_join_stats(source, target, dedict, resource) for
            source, target in zip(pairs['source'], pairs['target'])]
    )
    
    # ligand_ or receptor + stat_keys
    complex_cols = list(product(['ligand', 'receptor'], stat_names))
    complex_cols = [f'{x}_{y}' for x, y in complex_cols]
    
    lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                         _key_cols=_key_cols,
                                         expr_prop=expr_prop,
                                         return_all_lrs=return_all_lrs,
                                         complex_cols=complex_cols
                                        )
    lr_res = lr_res.drop(['prop_min', 'interaction'], axis=1)
    
    
    # summarise stats for each lr
    for key in stat_keys:
        stat_columns = lr_res.columns[lr_res.columns.str.endswith(key)]
        lr_res.loc[:, f'interaction_{key}'] = lr_res.loc[:, stat_columns].mean(axis=1)
    
    return lr_res
    
    

