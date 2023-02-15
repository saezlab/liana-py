import numpy as np
import pandas as pd
import anndata
from pandas import DataFrame

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.stats import norm
from tqdm import tqdm


def get_spatial_proximity(adata: anndata.AnnData,
                          parameter,
                          family='gaussian',
                          cutoff=None,
                          n_neighbors=None,
                          bypass_diagonal=False,
                          inplace=True):
    """
    Generate spatial proximity weights using Euclidean distance.

    Parameters
    ----------
    adata
        `AnnData` object with spatial coordinates (in 'spatial') in `adata.obsm`.
    parameter
         Denotes signaling length (`l`)
    family
        Functions used to generate proximity weights. The following options are available:
        ['gaussian', 'spatialdm', 'exponential', 'linear']
    cutoff
        Vales below this cutoff will be set to 0
    n_neighbors
        Find k nearest neighbours, use it as a proximity mask. In other words,
        only the proximity of the nearest neighbours is kept as calculated
        by the specified radial basis function, the remainder are set to 0.
    bypass_diagonal
        Logical, sets proximity diagonal to 0 if true.
    inplace
        If true return `DataFrame` with results, else assign to `.obsm`.

    Returns
    -------
    If ``inplace = False``, returns an `np.array` with spatial proximity weights.
    Otherwise, modifies the ``adata`` object with the following key:
        - :attr:`anndata.AnnData.obsm` ``['proximity']`` with the aforementioned array
    """

    families = ['gaussian', 'spatialdm', 'exponential', 'linear']
    if family not in families:
        raise AssertionError(f"{family} must be a member of {families}")

    if (cutoff is None) & (n_neighbors is None):
        raise ValueError("`cutoff` or `n_neighbors` must be provided!")

    assert 'spatial' in adata.obsm

    coordinates = pd.DataFrame(adata.obsm['spatial'],
                               index=adata.obs_names,
                               columns=['x', 'y'])

    proximity = pdist(coordinates, 'euclidean')
    proximity = squareform(proximity)

    # prevent overflow
    proximity = np.array(proximity, dtype=np.float64)
    parameter = np.array(parameter, dtype=np.float64)

    if family == 'gaussian':
        proximity = np.exp(-(proximity ** 2.0) / (2.0 * parameter ** 2.0))
    elif family == 'misty_rbf':
        proximity = np.exp(-(proximity ** 2.0) / (parameter ** 2.0))
    elif family == 'exponential':
        proximity = np.exp(-proximity / parameter)
    elif family == 'linear':
        proximity = 1 - proximity / parameter
        proximity[proximity < 0] = 0

    if bypass_diagonal:
        np.fill_diagonal(proximity, 0)

    if cutoff is not None:
        proximity[proximity < cutoff] = 0
    if n_neighbors is not None:
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(proximity)
        knn = nn.kneighbors_graph(proximity).toarray()
        proximity = proximity * knn  # knn works as mask

    spot_n = proximity.shape[0]
    assert spot_n == adata.shape[0]

    # speed up
    if spot_n > 1000:
        proximity = proximity.astype(np.float16)

    proximity = csr_matrix(proximity)

    adata.obsm['proximity'] = proximity
    return None if inplace else proximity


def _rename_means(lr_stats, entity):
    df = lr_stats.copy()
    df.columns = df.columns.map(lambda x: entity + '_' + str(x) if x != 'gene' else 'gene')
    return df.rename(columns={'gene': entity})


def _local_to_dataframe(idx, columns, array):
    return DataFrame(array, index=idx, columns=columns)


def _get_ordered_matrix(mat, pos, order):
    _indx = np.array([pos[x] for x in order])
    return mat[:, _indx].T


def _local_permutation_pvals(x_mat, y_mat, weight, local_truth, local_fun,n_perm, seed, positive_only, **kwargs):
    """
    Calculate local pvalues for a given local score function.

    Parameters
    ----------
    x_mat
        2D array with x variables
    y_mat
        2D array with y variables
    local_truth
        2D array with non-permuted local scores/co-expressions
    weight
        proximity weights
    n_perm
        number of permutations
    seed
        Reproducibility seed
    positive_only
        Whether to mask negative correlations pvalue

    Returns
    -------
    2D array with shape(n_spot, xy_n)

    """
    rng = np.random.default_rng(seed)

    xy_n = local_truth.shape[0]
    spot_n = local_truth.shape[1]
    
    # permutation cubes to be populated
    local_pvals = np.zeros((xy_n, spot_n))
    
    # shuffle the matrix
    for i in tqdm(range(n_perm)):
        _idx = rng.permutation(spot_n)
        perm_score = local_fun(x_mat = x_mat[_idx, :], y_mat=y_mat, weight=weight, **kwargs)
        if positive_only:
            local_pvals += np.array(perm_score >= local_truth, dtype=int)
        else:
            local_pvals += (np.array(np.abs(perm_score) >= np.abs(local_truth), dtype=int))

    local_pvals = local_pvals / n_perm

    ## TODO change this to directed which uses the categories as mask
    if positive_only:  # TODO change to directed mask (both, negative, positive)
        # only keep positive pvals where either x or y is positive
        pos_msk = ((x_mat > 0) + (y_mat > 0)).T
        local_pvals[~pos_msk] = 1

    return local_pvals


def _standardize_matrix(mat, local=True, axis=0):
    mat = np.array(mat - np.array(mat.mean(axis=axis)))
    if not local:
        mat = mat / np.sqrt(np.sum(mat ** 2, axis=axis, keepdims=True))
    return mat


def _encode_as_char(a):
    # if only positive
    if np.all(a >= 0):
        # TODO check if axis is correct
        a = _standardize_matrix(a, local=True, axis=0)
    a = np.where(a > 0, 'P', np.where(a < 0, 'N', 'Z'))
    return a


def _categorize(x, y):
    cat = np.core.defchararray.add(x, y)
    return cat


def _simplify_cats(df):
    """
    This function simplifies the categories of the co-expression matrix.
    
    Any combination of 'P' and 'N' is replaced by '-1' (negative co-expression).
    Any string containing 'Z' or 'NN' is replace by 0 (undefined or absence-absence)
    A 'PP' is replaced by 1 (positive co-expression)
    
    Note that  absence-absence is not definitive, but rather indicates that the 
    co-expression is between two genes expressed lower than their means
    """
    
    return df.replace({r'(^*Z*$)': 0, 'NN': 0, 'PP': 1, 'PN': -1, "NP": -1})

    
    
### Specific to SpatialDM - TODO generalize these functions
def _global_permutation_pvals(x_mat, y_mat, weight, global_r, n_perm, positive_only, seed):
    """
    Calculate permutation pvals

    Parameters
    ----------
    x_mat
        Matrix with x variables
    y_mat
        Matrix with y variables
    dist
        Proximity weights 2D array
    global_r
        Global Moran's I, 1D array
    n_perm
        Number of permutations
    positive_only
        Whether to mask negative p-values
    seed
        Reproducibility seed

    Returns
    -------
    1D array with same size as global_r

    """
    assert isinstance(weight, csr_matrix)
    rng = np.random.default_rng(seed)

    # initialize mat /w n_perm * number of X->Y
    idx = x_mat.shape[1]

    # permutation mat /w n_perms x LR_n
    perm_mat = np.zeros((n_perm, global_r.shape[0]))

    for perm in tqdm(range(n_perm)):
        _idx = rng.permutation(idx)
        perm_mat[perm, :] = ((x_mat[:, _idx] @ weight) * y_mat).sum(axis=1) # flipped x_mat

    if positive_only:
        global_pvals = 1 - (global_r > perm_mat).sum(axis=0) / n_perm
    else:
        # TODO Proof this makes sense
        global_pvals = 2 * (1 - (np.abs(global_r) > np.abs(perm_mat)).sum(axis=0) / n_perm)

    return global_pvals



def _global_zscore_pvals(weight, global_r, positive_only):
    """

    Parameters
    ----------
    weight
        proximity weight matrix (spot_n x spot_n)
    global_r
        Array with
    positive_only: bool
        whether to mask negative correlation p-values

    Returns
    -------
        1D array with the size of global_r

    """
    weight = np.array(weight.todense())
    spot_n = weight.shape[0]

    # global distance/weight variance as in spatialDM
    numerator = spot_n ** 2 * ((weight * weight).sum()) - \
                (2 * spot_n * (weight @ weight).sum()) + \
                (weight.sum() ** 2)
    denominator = spot_n ** 2 * (spot_n - 1) ** 2
    weight_var_sq = (numerator / denominator) ** (1 / 2)

    global_zscores = global_r / weight_var_sq

    if positive_only:
        global_zpvals = norm.sf(global_zscores)
    else:
        global_zpvals = norm.sf(np.abs(global_zscores)) * 2

    return global_zpvals




def _local_zscore_pvals(x_mat, y_mat, local_truth, weight, positive_only):
    """

    Parameters
    ----------
    x_mat
        2D array with x variables
    y_mat
        2D array with y variables
    local_r
        2D array with Local Moran's I
    weight
        proximity weights
    positive_only
        Whether to mask negative correlations pvalue

    Returns
    -------
    2D array of p-values with shape(n_spot, xy_n)

    """
    spot_n = weight.shape[0]

    x_norm = np.apply_along_axis(norm.fit, axis=0, arr=x_mat)
    y_norm = np.apply_along_axis(norm.fit, axis=0, arr=y_mat)

    # get x,y std
    x_sigma, y_sigma = x_norm[1, :], y_norm[1, :]

    x_sigma = x_sigma * spot_n / (spot_n - 1)
    y_sigma = y_sigma * spot_n / (spot_n - 1)

    std = _get_local_var(x_sigma, y_sigma, weight, spot_n)
    local_zscores = local_truth / std

    if positive_only:
        local_zpvals = norm.sf(local_zscores)
        pos_msk = ((x_mat > 0) + (y_mat > 0)).T  # mask?
        local_zpvals[~pos_msk] = 1
    else:
        local_zpvals = norm.sf(np.abs(local_zscores))

    return local_zpvals


def _get_local_var(x_sigma, y_sigma, weight, spot_n):
    """
    Spatial weight variance as in spatialDM (Li et al., 2022)

    Parameters
    ----------
    x_sigma
        Standard deviations for each x (e.g. std of all ligands in the matrix)
    y_sigma
        Standard deviations for each y (e.g. std of all receptors in the matrix)
    weight
        proximity weight matrix
    spot_n
        number of spots/cells in the matrix

    Returns
    -------
    2D array of standard deviations with shape(n_spot, xy_n)

    """
    weight_sq = (np.array(weight.todense()) ** 2).sum(axis=1)

    n_weight = 2 * (spot_n - 1) ** 2 / spot_n ** 2
    sigma_prod = x_sigma * y_sigma
    core = n_weight * sigma_prod

    var = np.multiply.outer(weight_sq, core) + core
    std = var ** 0.5

    return std.T