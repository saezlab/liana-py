"""
Functions to deal with protein complexes
"""
from __future__ import annotations

import pandas as pd


def filter_reassemble_complexes(lr_res,
                                _key_cols,
                                complex_cols,
                                expr_prop,
                                return_all_lrs=False,
                                complex_policy='min'):
    """
    Reassemble complexes from exploded long-format pandas Dataframe.

    Parameters
    ----------
    lr_res
        long-format pandas dataframe with exploded complex subunits
    _key_cols
        primary key for lr_res, typically a list with the following elements -
        ['source', 'target', 'ligand_complex', 'receptor_complex']
    complex_cols
        method/complex-relevant columns
    expr_prop
        minimum expression proportion for each subunit in a complex
    return_all_lrs
        Bool whether to return all LRs, or only those that surpass the expr_prop
        threshold. `False` by default.
    complex_policy
        approach by which the complexes are reassembled

    Return
    -----------
    lr_res: a reduced long-format pandas dataframe
    """
    # Filter by expr_prop (inner join only complexes where all subunits are expressed)
    expressed = (lr_res[_key_cols + ['ligand_props', 'receptor_props']]
                 .set_index(_key_cols)
                 .stack()
                 .groupby(_key_cols)
                 .agg(prop_min=complex_policy)
                 .reset_index()
                 )
    expressed = expressed[expressed['prop_min'] >= expr_prop]

    if not return_all_lrs:
        lr_res = lr_res.merge(expressed, how='inner', on=_key_cols)
    else:
        expressed['lrs_to_keep'] = True
        lr_res = lr_res.merge(expressed, how='left', on=_key_cols)
        lr_res['lrs_to_keep'].fillna(value=False, inplace=True)

    # check if complex policy is only min
    aggs = {complex_policy, 'min'}

    cols_dict = {}
    for col in complex_cols:
        lr_res = _reduce_complexes(col=col, cols_dict=cols_dict,
                                   lr_res=lr_res, key_cols=_key_cols,
                                   aggs=aggs)

    return lr_res


def _reduce_complexes(col: str,
                      cols_dict: dict,
                      lr_res: pd.DataFrame,
                      key_cols: list,
                      aggs: (dict | str)
                      ):
    """
    Reduce the complexes

    Parameters
    ------------

    col
        column by which we are reducing
    cols_dict
        dictionary that we populate with the reduced results for each ligand and receptor column
    lr_res
     liana_pipe generated long DataFrame
    key_cols
        a list of columns that define each row as unique
    aggs
        dictionary with the way(s) by which we aggregate. Note 'min' should be there -
        we need the miniumum to find the lowest expression subunits, which are then used
        to reduce the exploded complexes

    Return
    -----------
    lr_res with exploded complexes reduced to only the minimum (default) subunit

    """
    # Group by keys
    lr_res = lr_res.groupby(key_cols)

    # Get min cols by which we will join - CHANGE WITH A FLAG INSTEAD !!!!
    # then rename from agg name to column name (e.g. 'min' to 'ligand_min')
    cols_dict[col] = lr_res[col].agg(aggs).reset_index().copy(). \
        rename(columns={agg: col.split('_')[0] + '_' + agg for agg in aggs})

    # right is the min subunit for that column
    join_key = col.split('_')[0] + '_min'  # ligand_min or receptor_min

    # Here, I join the min value and keep only those rows that match
    lr_res = lr_res.obj.merge(cols_dict[col], on=key_cols)
    lr_res = lr_res[lr_res[col] == lr_res[join_key]].drop(join_key, axis=1)

    return lr_res


def explode_complexes(resource: pd.DataFrame,
                      SOURCE='ligand',
                      TARGET='receptor') -> pd.DataFrame:
    """
    Function to explode ligand-receptor complexes

    Parameters
    ----------
    resource
        Ligand-receptor resource
    SOURCE
        Name of the source (typically ligand) column
    TARGET
        Name of the target (typically receptor) column

    Returns
    -------
    A resource with exploded complexes

    """
    resource['interaction'] = resource[SOURCE] + '&' + resource[TARGET]
    resource = (resource.set_index('interaction')
                .apply(lambda x: x.str.split('_'))
                .explode([TARGET])
                .explode(SOURCE)
                .reset_index()
                )
    resource[[f'{SOURCE}_complex', f'{TARGET}_complex']] = resource[
        'interaction'].str.split('&', expand=True)

    return resource
