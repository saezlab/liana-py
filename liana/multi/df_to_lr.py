import numpy as np
import pandas as pd
from itertools import product

from liana.method._pipe_utils import prep_check_adata, assert_covered, filter_resource, _check_groupby
from liana.method._pipe_utils._common import _join_stats, _get_props, _get_groupby_subset
from liana.resource import explode_complexes, filter_reassemble_complexes
from liana.resource.select_resource import _handle_resource

from liana._logging import _logg
from liana._docs import d
from liana._constants import DefaultValues as V, InternalValues as I, PrimaryColumns as P

@d.dedent
def df_to_lr(adata,
             dea_df,
             groupby,
             stat_keys,
             resource_name = V.resource_name,
             resource = V.resource,
             interactions = V.interactions,
             groupby_pairs=V.groupby_pairs,
             layer = V.layer,
             use_raw = V.layer,
             expr_prop = V.expr_prop,
             min_cells = V.min_cells,
             complex_col = None,
             return_all_lrs=V.return_all_lrs,
             source_labels = None,
             target_labels = None,
             lr_sep=V.lr_sep,
             verbose = V.verbose,
             ):
    """
    Convert DEA results to ligand-receptor pairs.

    Parameters
    ----------
    %(adata)s
    dea_df : pd.DataFrame
        DEA results. Index must match adata.var_names
    %(groupby)s
    stat_keys : list
        List of statistics to be used for ligand-receptor pairs
    %(resource_name)s
    %(resource)s
    %(interactions)s
    %(layer)s
    %(use_raw)s
    %(expr_prop)s
    %(min_cells)s
    complex_col : str, optional
        Column in `dea_df` to use for complex expression. Default is None.
        If None, will use mean expression ('expr') calculated per group in `groupby`.
    %(return_all_lrs)s
    %(source_labels)s
    %(target_labels)s
    %(lr_sep)s
    %(verbose)s

    Returns
    -------
    Returns a pd.DataFrame with joined ligand-receptor pairs and statistics.

    """
    _check_groupby(adata=adata, groupby=groupby, verbose=verbose)
    if (groupby not in adata.obs.columns) or (groupby not in dea_df.columns):
        raise ValueError('groupby must match a column in both adata.obs and dea_df')
    if not np.any(adata.var_names.isin(dea_df.index)):
        raise ValueError('index of dea_df must match adata.var_names')
    if len(np.intersect1d(adata.obs[groupby].unique(), dea_df[groupby].unique())) == 0:
        raise AssertionError("`groupby` intersect between `dea_df` and `adata` is 0. Please check `groupby`.")

    resource = _handle_resource(interactions=interactions,
                                resource=resource,
                                resource_name=resource_name,
                                verbose=verbose)

    stat_names = ['expr', 'props'] + stat_keys
    if complex_col is not None:
        if complex_col not in stat_names:
            raise ValueError(f'complex_col must be one of `stat_keys`:{stat_keys} or the stats calculated by default: {stat_names}!')
        stat_names = stat_names[stat_names.index(complex_col):] + stat_names[:stat_names.index(complex_col)]
    else:
        complex_col = 'expr'

    groupby_subset = _get_groupby_subset(groupby_pairs=groupby_pairs)

    # Check and Reformat Mat if needed
    adata = prep_check_adata(adata=adata,
                             groupby=groupby,
                             groupby_subset=groupby_subset,
                             min_cells=min_cells,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose
                             )

    # reduce dim of adata
    intersect = np.intersect1d(adata.var_names, dea_df.index)
    if intersect.shape[0]==adata.shape[1]:
        _logg('Features in adata and dea_df are mismatched.', verbose=verbose, level='warn')
    adata =  adata[:, intersect]

    # get label cats
    labels = adata.obs[I.label].cat.categories
    dedict = {}
    for label in labels:
        temp = adata[adata.obs[I.label] == label, :]
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

    if groupby_pairs is not None:
        pairs = pairs.merge(groupby_pairs, on=[P.source, P.target], how='inner')

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
                                         _key_cols=P.primary,
                                         expr_prop=expr_prop,
                                         return_all_lrs=return_all_lrs,
                                         complex_cols=_placeholders
                                        )
    lr_res = lr_res.drop(['prop_min', 'interaction', *_placeholders], axis=1)

    # summarise stats for each lr
    for key in stat_names:
        stat_columns = lr_res.columns[lr_res.columns.str.endswith(key)]
        lr_res.loc[:, f'interaction_{key}'] = lr_res.loc[:, stat_columns].mean(axis=1)

    lr_res['interaction'] = lr_res['ligand_complex'] + lr_sep + lr_res['receptor_complex']

    return lr_res
