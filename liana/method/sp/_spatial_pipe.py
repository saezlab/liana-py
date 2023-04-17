from __future__ import annotations

import numpy as np
import pandas as pd
import anndata
from anndata import AnnData
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
                          inplace=True
                          ):
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

    dist = pdist(coordinates, 'euclidean')
    dist = squareform(dist)

    # prevent overflow
    dist = np.array(dist, dtype=np.float64)
    parameter = np.array(parameter, dtype=np.float64)

    if family == 'gaussian':
        proximity = np.exp(-(dist ** 2.0) / (2.0 * parameter ** 2.0))
    elif family == 'misty_rbf':
        proximity = np.exp(-(dist ** 2.0) / (parameter ** 2.0))
    elif family == 'exponential':
        proximity = np.exp(-dist / parameter)
    elif family == 'linear':
        proximity = 1 - dist / parameter
        proximity[proximity < 0] = 0
    else:
        raise ValueError("Please specify a valid family to generate proximity weights")

    if bypass_diagonal:
        np.fill_diagonal(proximity, 0)

    if cutoff is not None:
        proximity[proximity < cutoff] = 0
    if n_neighbors is not None:
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(proximity)
        knn = nn.kneighbors_graph(proximity).toarray()
        proximity = proximity * knn  # knn works as a mask

    spot_n = proximity.shape[0]
    assert spot_n == adata.shape[0]

    # speed up
    if spot_n > 1000:
        proximity = proximity.astype(np.float16)

    proximity = csr_matrix(proximity)

    adata.obsp['proximity'] = proximity
    return None if inplace else proximity


def _rename_means(lr_stats, entity):
    df = lr_stats.copy()
    df.columns = df.columns.map(lambda x: entity + '_' + str(x) if x != 'gene' else 'gene')
    return df.rename(columns={'gene': entity})


def _local_to_dataframe(array, idx, columns):
    if array is None:
        return None
    return DataFrame(array.T, index=idx, columns=columns)


def _get_ordered_matrix(mat, pos, order):
    _indx = np.array([pos[x] for x in order])
    return mat[:, _indx].T


def _local_permutation_pvals(x_mat, y_mat, weight, local_truth, local_fun, n_perms, seed,
                             positive_only, **kwargs):
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
    n_perms
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
    for i in tqdm(range(n_perms)):
        _idx = rng.permutation(spot_n)
        perm_score = local_fun(x_mat=x_mat[_idx, :], y_mat=y_mat, weight=weight, **kwargs)
        if positive_only:
            local_pvals += np.array(perm_score >= local_truth, dtype=int)
        else:
            local_pvals += (np.array(np.abs(perm_score) >= np.abs(local_truth), dtype=int))

    local_pvals = local_pvals / n_perms

    ## TODO change this to directed which uses the categories as mask
    if positive_only:  # TODO change to directed mask (both, negative, positive)
        # only keep positive pvals where either x or y is positive
        pos_msk = ((x_mat > 0) + (y_mat > 0)).T
        local_pvals[~pos_msk] = 1

    return local_pvals


# def _standardize_matrix(mat, local=True, axis=0):
#     mat = np.array(mat - np.array(mat.mean(axis=axis)))
#     if not local:
#         mat = mat / np.sqrt(np.sum(mat ** 2, axis=axis, keepdims=True))
#     return mat


def _standardize_matrix(mat, local=True, axis=0):
    spot_n = mat.shape[1]
    
    if not local:
        spot_n = 1
    
    mat = np.array(mat - np.array(mat.mean(axis=axis)))
    mat = mat / np.sqrt(np.sum(mat ** 2, axis=axis, keepdims=True) / spot_n)
    
    return mat


def _encode_as_char(a, weight):
    # if only positive
    if np.all(a >= 0):
        a = _standardize_matrix(a, local=True, axis=0)
    
    # to get a sign for each spot, we multiply by proximities 
    a = a @ weight
    
    a = np.where(a > 0, 'P', np.where(a < 0, 'N', 'Z'))
    return a


def _categorize(x_mat, y_mat, weight, idx, columns):
    x_cats = _encode_as_char(x_mat.A, weight)
    y_cats = _encode_as_char(y_mat.A, weight)
    
    # add the two categories, and simplify them to ints
    cats = np.core.defchararray.add(x_cats, y_cats)
    cats = _simplify_cats(cats)
    
    cats = _local_to_dataframe(array=cats,
                               idx=idx,
                               columns=columns
                               )
    
    return cats


def _simplify_cats(cats):
    """
    This function simplifies the categories of the co-expression matrix.
    
    Any combination of 'P' and 'N' is replaced by '-1' (negative co-expression).
    Any string containing 'Z' or 'NN' is replace by 0 (undefined or absence-absence)
    A 'PP' is replaced by 1 (positive co-expression)
    
    Note that  absence-absence is not definitive, but rather indicates that the 
    co-expression is between two genes expressed lower than their means
    """
    cats = np.char.replace(cats, 'PP', '1')
    cats = np.char.replace(cats, 'PN', '-1')
    cats = np.char.replace(cats, 'NP', '-1')
    msk = (cats!='1') * (cats!='-1')
    cats[msk] = '0'
    
    return cats.astype(int)
    

def _global_permutation_pvals(x_mat, y_mat, weight, global_r, n_perms, positive_only, seed):
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
    n_perms
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

    # initialize mat /w n_perms * number of X->Y
    idx = x_mat.shape[1]

    # permutation mat /w n_permss x LR_n
    perm_mat = np.zeros((n_perms, global_r.shape[0]))

    for perm in tqdm(range(n_perms)):
        _idx = rng.permutation(idx)
        perm_mat[perm, :] = ((x_mat[:, _idx] @ weight) * y_mat).sum(axis=1)  # flipped x_mat

    if positive_only:
        global_pvals = 1 - (global_r > perm_mat).sum(axis=0) / n_perms
    else:
        # TODO Proof this makes sense
        global_pvals = 2 * (1 - (np.abs(global_r) > np.abs(perm_mat)).sum(axis=0) / n_perms)

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


def _global_spatialdm(x_mat,
                      y_mat,
                      weight,
                      seed,
                      n_perms,
                      pvalue_method,
                      positive_only
                      ):
    """
    Global Moran's Bivariate I as implemented in SpatialDM

    Parameters
    ----------
    x_mat
        Gene expression matrix for entity x (e.g. ligand)
    y_mat
        Gene expression matrix for entity y (e.g. receptor)
    x_pos
        Index positions of entity x (e.g. ligand) in `mat`
    y_pos
        Index positions of entity y (e.g. receptor) in `mat`
    xy_dataframe
        a dataframe with x,y relationships to be estimated, for example `lr_res`.
    weight
        proximity weight matrix, obtained e.g. via `liana.method.get_spatial_proximity`.
        Note that for spatialDM/Morans'I `weight` has to be weighed by n / sum(W).
    seed
        Reproducibility seed
    n_perms
        Number of permutatins to perform (if `pvalue_method`=='permutation')
    pvalue_method
        Method to estimate pseudo p-value, must be in ['permutation', 'analytical']
    positive_only
        Whether to return only p-values for positive spatial correlations.
        By default, `True`.

    Returns
    -------
    Tupple of 2 1D Numpy arrays of size xy_dataframe.shape[1],
    or in other words calculates global_I and global_pval for
    each interaction in `xy_dataframe`

    """
    # Get global r
    global_r = ((x_mat @ weight) * y_mat).sum(axis=1)

    # calc p-values
    if pvalue_method == 'permutation':
        global_pvals = _global_permutation_pvals(x_mat=x_mat,
                                                 y_mat=y_mat,
                                                 weight=weight,
                                                 global_r=global_r,
                                                 n_perms=n_perms,
                                                 positive_only=positive_only,
                                                 seed=seed
                                                 )
    elif pvalue_method == 'analytical':
        global_pvals = _global_zscore_pvals(weight=weight,
                                            global_r=global_r,
                                            positive_only=positive_only)
    elif pvalue_method is None:
        global_pvals = None

    return np.array((global_r, global_pvals))


def _run_scores_pipeline(xy_stats, x_mat, y_mat, idx, local_fun,
                         weight, pvalue_method, positive_only, n_perms, seed):
    """
        Calculates local and global scores for each interaction in `xy_dataframe`

        Parameters
        ----------
        xy_stats
            a dataframe with x,y relationships to be estimated, for example `lr_res`.
        x_mat
            Gene expression matrix for entity x (e.g. ligand)
        y_mat
            Gene expression matrix for entity y (e.g. receptor)
        idx
            Index positions of cells/spots (i.e. adata.obs.index)
        local_fun
            Function to calculate local scores, e.g. `liana.method._local_morans`
        weight
            proximity weight matrix, obtained e.g. via `liana.method.get_spatial_proximity`.
            Note that for spatialDM/Morans'I `weight` has to be weighed by n / sum(W).
        pvalue_method
            Method to estimate pseudo p-value, must be in ['permutation', 'analytical']
        positive_only
            Whether to return only p-values for positive spatial correlations.
            By default, `True`.
        n_perms
            Number of permutatins to perform (if `pvalue_method`=='permutation')
        seed
            Reproducibility seed

        Returns
        -------
        A dataframe and two 2D arrays of size xy_dataframe.shape[1], adata.shape[0]

        """
    local_scores, local_pvals = _get_local_scores(x_mat=x_mat.T,
                                                  y_mat=y_mat.T,
                                                  local_fun=local_fun,
                                                  weight=weight,
                                                  seed=seed,
                                                  n_perms=n_perms,
                                                  pvalue_method=pvalue_method,
                                                  positive_only=positive_only,
                                                  )

    # global scores fun
    xy_stats = _get_global_scores(xy_stats=xy_stats,
                                  x_mat=x_mat,
                                  y_mat=y_mat,
                                  local_fun=local_fun,
                                  pvalue_method=pvalue_method,
                                  weight=weight,
                                  seed=seed,
                                  n_perms=n_perms,
                                  positive_only=positive_only,
                                  local_scores=local_scores,
                                  )

    # convert to DataFrames
    local_scores = _local_to_dataframe(array=local_scores,
                                       idx=idx,
                                       columns=xy_stats['interaction'])
    if local_pvals is not None:
        local_pvals = _local_to_dataframe(array=local_pvals,
                                          idx=idx,
                                          columns=xy_stats['interaction'])

    return xy_stats, local_scores, local_pvals


def _get_local_scores(x_mat,
                      y_mat,
                      local_fun,
                      weight,
                      n_perms,
                      seed,
                      pvalue_method,
                      positive_only,
                      ):
    """
    Local Moran's Bivariate I as implemented in SpatialDM

    Parameters
    ----------
    x_mat
        Matrix with x variables
    y_mat
        Matrix with y variables
    x_pos
        Index positions of entity x (e.g. ligand) in `mat`
    y_pos
        Index positions of entity y (e.g. receptor) in `mat`
    xy_dataframe
        a dataframe with x,y relationships to be estimated, for example `lr_res`.
    weight
        proximity weight matrix, obtained e.g. via `liana.method.get_spatial_proximity`.
        Note that for spatialDM/Morans'I `weight` has to be weighed by n / sum(W).
    seed
        Reproducibility seed
    n_perms
        Number of permutations to perform (if `pvalue_method`=='permutation')
    pvalue_method
        Method to estimate pseudo p-value, must be in ['permutation', 'analytical']
    positive_only
        Whether to return only p-values for positive spatial correlations.
        By default, `True`.

    Returns
    -------
        Tupple of two 2D Numpy arrays of size (n_spots, n_xy),
         or in other words calculates local_I and local_pval for
         each interaction in `xy_dataframe` and each sample in mat
    """

    if local_fun.__name__ == '_local_morans':
        x_mat = _standardize_matrix(x_mat, local=True, axis=0)
        y_mat = _standardize_matrix(y_mat, local=True, axis=0)
        
        # NOTE: spatialdm do this, and also use .raw by default
        # x_mat = x_mat / np.max(x_mat, axis=0)
        # y_mat = y_mat / np.max(y_mat, axis=0)
        
    else:
        x_mat = x_mat.A
        y_mat = y_mat.A

    local_scores = local_fun(x_mat, y_mat, weight)

    if pvalue_method == 'permutation':
        local_pvals = _local_permutation_pvals(x_mat=x_mat,
                                               y_mat=y_mat,
                                               weight=weight,
                                               local_truth=local_scores,
                                               local_fun=local_fun,
                                               n_perms=n_perms,
                                               seed=seed,
                                               positive_only=positive_only
                                               )
    elif pvalue_method == 'analytical':
        local_pvals = _local_zscore_pvals(x_mat=x_mat,
                                          y_mat=y_mat,
                                          local_truth=local_scores,
                                          weight=weight,
                                          positive_only=positive_only
                                          )
    elif pvalue_method is None:
        local_pvals = None

    return local_scores, local_pvals


def _get_global_scores(xy_stats, x_mat, y_mat, local_fun, weight, pvalue_method, positive_only,
                       n_perms, seed, local_scores):
    if local_fun.__name__ == "_local_morans":
        xy_stats.loc[:, ['global_r', 'global_pvals']] = \
            _global_spatialdm(x_mat=_standardize_matrix(x_mat, local=False, axis=1),
                              y_mat=_standardize_matrix(y_mat, local=False, axis=1),
                              weight=weight,
                              seed=seed,
                              n_perms=n_perms,
                              pvalue_method=pvalue_method,
                              positive_only=positive_only
                              ).T
    else:
        # any other local score
        xy_stats.loc[:, ['global_mean', 'global_sd']] = np.vstack(
            [np.mean(local_scores, axis=1), np.std(local_scores, axis=1)]
            ).T

    return xy_stats


def _proximity_to_weight(proximity, local_fun):
    ## TODO add tests for this
    proximity = csr_matrix(proximity, dtype=np.float32)
    
    if local_fun.__name__ == "_local_morans":
        norm_factor = proximity.shape[0] / proximity.sum()
        proximity = norm_factor * proximity
        
        return csr_matrix(proximity)
    
    elif (proximity.shape[0] < 5000) | local_fun.__name__.__contains__("masked"):
    # NOTE vectorized is faster with non-sparse, masked_scores don't work with sparse
            return proximity.A
    else:
        return csr_matrix(proximity)


def _handle_proximity(adata, proximity, proximity_key):
    if proximity is None:
        if adata.obsp[proximity_key] is None:
            raise ValueError(f'No proximity matrix founds in mdata.obsp[{proximity_key}]')
        proximity = adata.obsp[proximity_key]
    proximity = csr_matrix(proximity, dtype=np.float32)
    return proximity
