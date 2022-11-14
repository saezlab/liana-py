import numpy as np
import pandas as pd
from functools import reduce
from scipy.stats import rankdata, beta


def _aggregate(lrs: dict,
               consensus,
               aggregate_method: str = 'rra',
               consensus_opts: list = None,
               _key_cols: list = None
               ) -> pd.DataFrame:
    """

    Parameters
    ---------
    lrs
        a list with results for all methods
    consensus
        ConsensusClass instance used to generate the lr results
    _key_cols
        should represent unique LRs columns by which to join
    aggregate_method
        method by which we aggregate the ranks. Options are ['rra', 'mean'],
        where 'rra' corresponds to the RRA method, 'mean' is just the mean of the ranks.
    consensus_opts
        consensus ranks to be obtained

    Returns
    -------
    A long pd.DataFrame with ranked LRs
    """

    # join the sc to the whole universe between the methods
    if _key_cols is None:
        _key_cols = ['source', 'target', 'ligand_complex', 'receptor_complex']
    if consensus_opts is None:
        consensus_opts = ['Steady', 'Magnitude', 'Specificity']

    lrs = [lrs[method].drop_duplicates(keep='first') for method in lrs]
    # reduce to a df with the shared keys + all relevant sc
    lr_res = reduce(
        lambda left, right:
        pd.merge(left, right, how='outer', on=_key_cols,
                 suffixes=('', '_duplicated')), lrs
    )
    # drop duplicated columns
    lr_res = lr_res.loc[:, ~lr_res.columns.str.endswith('_duplicated')]

    order_col = ''
    if 'Magnitude' in consensus_opts:
        lr_res[consensus.magnitude] = _rank_aggregate(lr_res.copy(),
                                                      consensus.magnitude_specs,
                                                      _key_cols,
                                                      aggregate_method=aggregate_method)
        order_col = consensus.magnitude
    if 'Specificity' in consensus_opts:
        lr_res[consensus.specificity] = _rank_aggregate(lr_res.copy(),
                                                        consensus.specificity_specs,
                                                        _key_cols,
                                                        aggregate_method=aggregate_method)
        order_col = consensus.specificity
    if 'Steady' in consensus_opts:
        lr_res[consensus.steady] = _rank_aggregate(lr_res.copy(),
                                                   consensus.steady_specs,
                                                   _key_cols,
                                                   aggregate_method=aggregate_method)
        order_col = consensus.steady

    lr_res = lr_res.sort_values(order_col)

    return lr_res


def _rank_aggregate(lr_res, specs, _key_cols, aggregate_method) -> np.array:
    """
    Aggregate method ranks

    Parameters
    ----------
    lr_res
        joined results from all methods
    specs
        specs dictionary where method_name:(score_name, score_desc)
    _key_cols
        columns by which we join
    aggregate_method
        method by which to aggregate the ranks

    Returns
    -------
    An array of values /w length of lr_res.shape[0]

    """
    assert aggregate_method in ['rra', 'mean']

    # Convert specs columns to ranks
    for spec in specs:
        score_name = specs[spec][0]
        ascending = specs[spec][1]

        if ascending:
            lr_res[score_name] = rankdata(lr_res[score_name], method='average')
        else:
            lr_res[score_name] = rankdata(lr_res[score_name] * -1,
                                          method='average')

    # get only the relevant ranks as a mat (joins order the keys)
    scores = list({specs[s][0] for s in specs})
    rmat = lr_res[scores].values

    if aggregate_method == 'rra':
        return _robust_rank_aggregate(rmat)
    elif aggregate_method == 'mean':
        return np.mean(rmat, axis=1)


def _corr_beta_pvals(p, k) -> np.array:
    """
    Correct beta p-values

    Parameters
    ----------
    p
        (min) p-value
    k
        total number of rows

    Returns
    -------
    An array with corrected p-values
    """
    p = np.clip(p * k, a_min=0, a_max=1)
    return p


def _rho_scores(rmat, dist_a, dist_b):
    """
    Calculate Beta Distribution Rho Scores
    
    ----------
    rmat
        a matrix where rows are the ranks/n for each interaction, while
        columns correspond to each method
    dist_a
        non-negative shape param a
    dist_b
        non-negative shape param b

    Returns
    -------
    A vector of pvals as implemented in the RRA method
    """

    # Sort values by sources (rows)
    rmat = np.sort(rmat, axis=1)
    # Calc beta cdf across rows
    p = beta.cdf(rmat, dist_a, dist_b)
    # get min pval per row
    p = np.min(p, axis=1)
    # correct p-vals
    rho = _corr_beta_pvals(p, k=rmat.shape[1])

    return rho


def _robust_rank_aggregate(rmat) -> np.array:
    """
    Calculate Robust Rank Aggregate as in Kolde et al., 2012

    Parameters
    ----------
    rmat
        Matrix with interaction ranks (rows) for each method (columns)

    Returns
    -------
    An array with p-values for each row
    """

    # 0-1 values depending on relative rank of
    # each interaction divided by the max of each method
    # due to max diffs due to ties
    rmat = rmat / np.max(rmat, axis=0)
    # generate dist_a/b with same row size as rmat
    dist_a = np.repeat([np.arange(rmat.shape[1])], rmat.shape[0], axis=0) + 1
    dist_b = rmat.shape[1] - dist_a + 1

    return _rho_scores(rmat, dist_a, dist_b)
