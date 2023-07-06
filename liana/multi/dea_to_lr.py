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
              complex_col=None,
              return_all_lrs=False,
              source_labels = None,
              target_labels = None,
              verbose = False,
              ):
    _key_cols = ['source', 'target', 'ligand_complex', 'receptor_complex']
        
    if (groupby not in adata.obs.columns) or (groupby not in dea_df.columns):
        raise ValueError(f'groupby must match a column in adata.obs and dea_df')
    if resource is None:
        if resource_name is not None:
            resource = select_resource(resource_name)
        else:
            raise ValueError('Please provide a `resource` or a valid `resource_name`')
    if any('_' in key for key in stat_keys):
        raise ValueError('stat_keys must not contain "_"')
    
    stat_names = ['expr', 'props'] + stat_keys
    if complex_col is not None:
        if complex_col not in stat_names:
            raise ValueError(f'complex_col must be one of {stat_names}')
        # set complex_col as first column
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
    complex_cols = list(product(['ligand', 'receptor'], [complex_col]))
    complex_cols = [f'{x}_{y}' for x, y in complex_cols]
    
    # assign receptor and ligand absolutes, NOTE to refactor this
    lr_res[['ligand_absolute', 'receptor_absolute']] = \
        lr_res[complex_cols].apply(lambda x: x.abs())
    
    lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                         _key_cols=_key_cols,
                                         expr_prop=expr_prop,
                                         return_all_lrs=return_all_lrs,
                                         complex_cols=['ligand_absolute', 'receptor_absolute']
                                        )
    lr_res = lr_res.drop(['prop_min', 'interaction', 'ligand_absolute', 'receptor_absolute'], axis=1)
    
    # summarise stats for each lr
    for key in stat_keys:
        stat_columns = lr_res.columns[lr_res.columns.str.endswith(key)]
        lr_res.loc[:, f'interaction_{key}'] = lr_res.loc[:, stat_columns].mean(axis=1)
    
    return lr_res
