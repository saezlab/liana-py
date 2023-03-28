import numpy as np
from scipy.stats import gmean

from liana.method._Method import Method, MethodMeta


# Internal Function to calculate Geometric LR_mean and p-values
def _gmean_score(x, perm_stats) -> tuple:
    """
    Calculate CellPhoneDB-like LR means and p-values

    Parameters
    ----------
    x
        DataFrame with LIANA results
    perm_stats
        Permutation statistics (2 (ligand-receptor), n_perms (number of permutations, n_rows in lr_res)

    Returns
    -------
    A tuple with lr_mean and p-value for x

    """
    lr_gmean = gmean((x['ligand_means'].values, x['receptor_means'].values), axis=0)
    lr_perm_means = gmean(perm_stats, axis=0)
    
    # calculate p-values
    n_perms = perm_stats.shape[1]
    gmean_pvals = np.sum(np.greater_equal(lr_perm_means, lr_gmean), axis=0) / n_perms
    
    return lr_gmean, gmean_pvals


# Initialize Meta
_geometric_mean = MethodMeta(method_name="Geometric Mean",
                             complex_cols=["ligand_means", "receptor_means"],
                             add_cols=[],
                             fun=_gmean_score,
                             magnitude="lr_gmeans",
                             magnitude_ascending=False,
                             specificity="gmean_pvals",
                             specificity_ascending=True,
                             permute=True,
                             reference="CellPhoneDBv2's permutation approach applied to the "
                                       "geometric means of ligand-receptors' mean, as opposed to "
                                       "their arithmetic mean."
                             )

# Initialize callable Method instance
geometric_mean = Method(_SCORE=_geometric_mean)
