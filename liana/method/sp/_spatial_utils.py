import numpy as np
import pandas as pd
import anndata
from pandas import DataFrame

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
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


def _local_to_dataframe(idx, columns, array):
    return DataFrame(array, index=idx, columns=columns)



def _local_permutation_pvals(x_mat, y_mat, dist, local_truth, local_fun,n_perm, seed, positive_only, **kwargs):
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
    dist
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
    assert isinstance(dist, csr_matrix)

    xy_n = local_truth.shape[0]
    spot_n = local_truth.shape[1]
    
    # permutation cubes to be populated
    local_pvals = np.zeros((xy_n, spot_n))
    
    # shuffle the matrix
    for i in tqdm(range(n_perm)):
        _idx = rng.permutation(spot_n)
        perm_score = local_fun(x_mat = x_mat[_idx, :], y_mat=y_mat, dist=dist, **kwargs)
        if positive_only:
            local_pvals += np.array(perm_score >= local_truth, dtype=int)
        else:
            local_pvals += (np.array(np.abs(perm_score) >= np.abs(local_truth), dtype=int))

    local_pvals = local_pvals / n_perm

    if positive_only:  # TODO change to directed mask (both, negative, positive)
        # only keep positive pvals where either x or y is positive
        pos_msk = ((x_mat > 0) + (y_mat > 0)).T
        local_pvals[~pos_msk] = 1

    return local_pvals

