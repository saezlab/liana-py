"""
Functions to deal with protein complexes
"""
from __future__ import annotations

import pandas as pd
from liana._logging import _logg
from liana._docs import d

@d.dedent
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
    %(expr_prop)s
    %(return_all_lrs)s
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
         # deal with duplicated subunits
         # subunits that are not expressed might not represent the most relevant subunit
        lr_res.drop_duplicates(subset=_key_cols, inplace=True)
        lr_res['lrs_to_keep'].fillna(value=False, inplace=True)
        lr_res['prop_min'].fillna(value=0, inplace=True)

    # check if complex policy is only min
    aggs = {complex_policy, 'min'}

    for col in complex_cols:
        lr_res = _reduce_complexes(col=col,
                                   lr_res=lr_res,
                                   key_cols=_key_cols,
                                   aggs=aggs)

    # check if there are any duplicated subunits
    duplicate_mask = lr_res.duplicated(subset=_key_cols, keep=False)
    if duplicate_mask.any():
        # check if there are any non-equal subunit values
        if not lr_res[duplicate_mask].groupby(_key_cols)[complex_cols].transform(lambda x: x.duplicated(keep=False)).all().all():
            _logg('There were duplicated subunits in the complexes. ' +
                 'The subunits were reduced to only the minimum expression subunit. ' +
                 'However, there were subunits that were not the same within a complex. ',
                 level='warn')
        lr_res = lr_res.drop_duplicates(subset=_key_cols, keep='first')

    return lr_res


def _reduce_complexes(col: str,
                      lr_res: pd.DataFrame,
                      key_cols: list,
                      aggs: (dict | str)
                      ):
    lr_res = lr_res.groupby(key_cols)

    # Get min cols by which we will join
    # then rename from agg name to column name (e.g. 'min' to 'ligand_min')
    lr_min = lr_res[col].agg(aggs).reset_index().copy(). \
        rename(columns={agg: col.split('_')[0] + '_' + agg for agg in aggs})

    # right is the min subunit for that column
    join_key = col.split('_')[0] + '_min'  # ligand_min or receptor_min

    # Here, I join the min value and keep only those rows that match
    lr_res = lr_res.obj.merge(lr_min, on=key_cols, how='inner')
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
