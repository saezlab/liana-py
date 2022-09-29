
def reassemble_complexes(lr_res, key_cols, complex_cols, complex_policy='min'):
    """
    Reassemble complexes from exploded long-format pandas Dataframe.

    Parameters
    ----------
    :param lr_res: a long-format pandas dataframe with exploded complex subunits
    :param key_cols: primary key for lr_res, typically a list with the following
    elements - ['source', 'target', 'ligand_complex', 'receptor_complex']
    :param complex_cols: method/complex-relevant columns
    :param complex_policy: approach by which the complexes are reassembled

    Return
    -----------
    :return: lr_res: a reduced long-format pandas dataframe
    """
    # check if complex policy is only min
    aggs = {complex_policy, 'min'}

    cols_dict = {}
    for col in complex_cols:
        lr_res = _reduce_complexes(col, cols_dict, lr_res, key_cols, aggs)

    return lr_res


# Function to reduce the complexes
def _reduce_complexes(col, cols_dict, lr_res, key_cols, aggs):
    # Group by keys
    lr_res = lr_res.groupby(key_cols)

    # Get min cols by which we will join
    # then rename from agg name to column name (e.g. 'min' to 'ligand_min')
    cols_dict[col] = lr_res[col].agg(aggs).reset_index().copy(). \
        rename(columns={agg: col.split('_')[0] + '_' + agg for agg in aggs})

    # left is lr_res /w the actual column name
    left_on = key_cols + [col]
    # right is the min subunit for that column
    join_key = col.split('_')[0] + '_min'  # ligand_min or receptor_min
    right_on = key_cols + [join_key]

    # Here, I join the min value and keep only those rows that match
    lr_res = lr_res.obj.merge(cols_dict[col], left_on=left_on,
                              right_on=right_on).drop(join_key, 1)

    return lr_res
