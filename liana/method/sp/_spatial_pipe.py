from __future__ import annotations

import numpy as np
from pandas import Series

from scipy.sparse import csr_matrix, isspmatrix_csr, hstack
from scipy.stats import norm
from tqdm import tqdm
from anndata import AnnData

def _rename_means(lr_stats, entity):
    df = lr_stats.copy()
    df.columns = df.columns.map(lambda x: entity + '_' + str(x) if x != 'gene' else 'gene')
    return df.rename(columns={'gene': entity})


def _local_permutation_pvals(x_mat, y_mat, weight,
                             local_truth, local_fun,
                             n_perms, seed,
                             mask_negatives, verbose):
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
        connectivity weights
    n_perms
        number of permutations
    seed
        Reproducibility seed
    mask_negatives
        Whether to mask negative correlations p-value

    Returns
    -------
    2D array with shape(n_spot, xy_n)

    """
    rng = np.random.default_rng(seed)

    spot_n = local_truth.shape[0]
    xy_n = local_truth.shape[1]

    # permutation cubes to be populated
    local_pvals = np.zeros((spot_n, xy_n))

    # shuffle the matrix
    for i in tqdm(range(n_perms), disable=not verbose):
        _idx = rng.permutation(spot_n)
        perm_score = local_fun(x_mat=x_mat[_idx, :], y_mat=y_mat[_idx, :], weight=weight)
        if mask_negatives:
            local_pvals += np.array(perm_score >= local_truth, dtype=int)
        else:
            local_pvals += np.array(np.abs(perm_score) >= np.abs(local_truth), dtype=int)

    local_pvals = local_pvals / n_perms

    return local_pvals


def _zscore(mat, local=True, axis=0):
    spot_n = mat.shape[axis]

    # NOTE: specific to global SpatialDM
    if not local:
        spot_n = 1

    mat = mat - mat.mean(axis=axis)
    mat = mat / np.sqrt(np.sum(np.power(mat, 2), axis=axis) / spot_n)

    return np.array(mat)


def _encode_cats(a, weight):
    if np.all(a >= 0): # NOTE: this should work with sparse matrices!!
        a = _zscore(a)
    a = weight @ a
    a = np.where(a > 0, 1, np.where(a < 0, -1, np.nan))
    return a

def _categorize(x_mat, y_mat, weight):
    # NOTE: this should work with sparse matrices!!
    x_cats = _encode_cats(x_mat.A, weight)
    y_cats = _encode_cats(y_mat.A, weight)

    # add the two categories, and simplify them to ints
    cats = x_cats + y_cats
    cats = np.where(cats == 2, 1, np.where(cats == 0, -1, 0))

    return cats


def _global_permutation_pvals(x_mat, y_mat, weight, global_r, n_perms, mask_negatives, seed, verbose):
    """
    Calculate permutation pvals

    Parameters
    ----------
    x_mat
        Matrix with x variables
    y_mat
        Matrix with y variables
    dist
        connectivity weights 2D array
    global_r
        Global Moran's I, 1D array
    n_perms
        Number of permutations
    mask_negatives
        Whether to mask negative p-values
    seed
        Reproducibility seed

    Returns
    -------
    1D array with same size as global_r

    """
    rng = np.random.default_rng(seed)

    # initialize mat /w n_perms * number of X->Y
    idx = x_mat.shape[0]

    # permutation mat /w n_permss x LR_n
    perm_mat = np.zeros((n_perms, global_r.shape[0]))

    for perm in tqdm(range(n_perms), disable=not verbose):
        _idx = rng.permutation(idx)
        perm_mat[perm, :] = _global_r(x_mat=x_mat[_idx, :],
                                      y_mat=y_mat[_idx, :],
                                      weight=weight)

    if mask_negatives:
        global_pvals = 1 - (global_r > perm_mat).sum(axis=0) / n_perms
    else:
        global_pvals = 1 - (np.abs(global_r) > np.abs(perm_mat)).sum(axis=0) / n_perms

    return global_pvals


def _global_zscore_pvals(weight, global_r, mask_negatives):
    """

    Parameters
    ----------
    weight
        connectivity weight matrix (spot_n x spot_n)
    global_r
        Array with
    mask_negatives: bool
        whether to mask negative correlation p-values

    Returns
    -------
        1D array with the size of global_r

    """
    if not isinstance(weight, np.ndarray):
        weight = np.array(weight.todense())
    spot_n = weight.shape[0]

    # global distance/weight variance as in spatialDM
    numerator = spot_n ** 2 * ((weight * weight).sum()) - \
                (2 * spot_n * (weight @ weight).sum()) + \
                (weight.sum() ** 2)
    denominator = spot_n ** 2 * (spot_n - 1) ** 2
    weight_var_sq = (numerator / denominator) ** (1 / 2)

    global_zscores = global_r / weight_var_sq

    if mask_negatives:
        global_zpvals = norm.sf(global_zscores)
    else:
        global_zpvals = norm.sf(np.abs(global_zscores)) * 2

    return global_zpvals


def _local_zscore_pvals(x_mat, y_mat, local_truth, weight, mask_negatives):
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
        connectivity weights
    mask_negatives
        Whether to mask negative correlations pvalue

    Returns
    -------
    2D array of p-values with shape(n_spot, xy_n)

    """
    spot_n = x_mat.shape[0]

    x_norm = np.apply_along_axis(norm.fit, axis=0, arr=x_mat)
    y_norm = np.apply_along_axis(norm.fit, axis=0, arr=y_mat)

    # get x,y std
    x_sigma, y_sigma = x_norm[1, :], y_norm[1, :]

    x_sigma = x_sigma * spot_n / (spot_n - 1)
    y_sigma = y_sigma * spot_n / (spot_n - 1)

    std = _get_local_var(x_sigma, y_sigma, weight, spot_n)
    local_zscores = local_truth / std

    if mask_negatives:
        local_zpvals = norm.sf(local_zscores)
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
        connectivity weight matrix
    spot_n
        number of spots/cells in the matrix

    Returns
    -------
    2D array of standard deviations with shape(n_spot, xy_n)

    """
    if not isinstance(weight, np.ndarray):
        weight = np.array(weight.todense())

    weight_sq = (weight ** 2).sum(axis=1)

    dim = 2 * (spot_n - 1) ** 2 / spot_n ** 2
    sigma_prod = x_sigma * y_sigma
    core = dim * sigma_prod

    var = np.multiply.outer(weight_sq, core) + core
    std = var ** 0.5

    return std

def _global_r(x_mat, y_mat, weight):
    return ((weight @ x_mat) * y_mat).sum(axis=0)

def _global_spatialdm(x_mat,
                      y_mat,
                      weight,
                      seed,
                      n_perms,
                      mask_negatives,
                      verbose
                      ):
    # Get global r
    global_r = _global_r(x_mat=x_mat, y_mat=y_mat, weight=weight)

    # calc p-values
    if n_perms is None:
        global_pvals = None
    elif n_perms > 0:
        global_pvals = _global_permutation_pvals(x_mat=x_mat,
                                                 y_mat=y_mat,
                                                 weight=weight,
                                                 global_r=global_r,
                                                 n_perms=n_perms,
                                                 mask_negatives=mask_negatives,
                                                 seed=seed,
                                                 verbose=verbose
                                                 )
    elif n_perms==0:
        global_pvals = _global_zscore_pvals(weight=weight,
                                            global_r=global_r,
                                            mask_negatives=mask_negatives)

    return global_r, global_pvals

def _norm_max(X, axis=0):
    X = X / X.max(axis=axis).A
    X = _zscore(X, local=True, axis=axis)
    X = np.where(np.isnan(X), 0, X)

    return X

def _get_local_scores(x_mat,
                      y_mat,
                      local_fun,
                      weight,
                      n_perms,
                      seed,
                      mask_negatives,
                      verbose
                      ):
    """
    Local Moran's Bivariate I as implemented in SpatialDM

    Returns
    -------
        Tupple of two 2D Numpy arrays of size (n_spots, n_xy),
         or in other words calculates local_I and local_pval for
         each interaction in `xy_dataframe` and each sample in mat
    """

    if local_fun.__name__ == '_local_morans':
        x_mat = _norm_max(x_mat)
        y_mat = _norm_max(y_mat)
    else:
        x_mat = x_mat.A
        y_mat = y_mat.A

    local_scores = local_fun(x_mat, y_mat, weight)

    if n_perms is None:
        local_pvals = None
    elif n_perms > 0:
        local_pvals = _local_permutation_pvals(x_mat=x_mat,
                                               y_mat=y_mat,
                                               weight=weight,
                                               local_truth=local_scores,
                                               local_fun=local_fun,
                                               n_perms=n_perms,
                                               seed=seed,
                                               mask_negatives=mask_negatives,
                                               verbose=verbose
                                               )
    elif n_perms == 0:
        local_pvals = _local_zscore_pvals(x_mat=x_mat,
                                          y_mat=y_mat,
                                          local_truth=local_scores,
                                          weight=weight,
                                          mask_negatives=mask_negatives
                                          )

    return local_scores, local_pvals


def _add_complexes_to_var(adata, entities, complex_sep='_'):
    """
    Generate an AnnData object with complexes appended as variables.
    """
    complexes = entities[Series(entities).str.contains(complex_sep)]

    X = None

    for comp in complexes:
        subunits = comp.split(complex_sep)

        # keep only complexes, the subunits of which are in var
        if all([subunit in adata.var.index for subunit in subunits]):
            adata.var.loc[comp, :] = None

            # create matrix for this complex
            new_array = csr_matrix(adata[:, subunits].X.min(axis=1))

            if X is None:
                X = new_array
            else:
                X = hstack((X, new_array))

    adata = AnnData(X=hstack((adata.X, X)),
                    obs=adata.obs, var=adata.var,
                    obsm=adata.obsm, obsp=adata.obsp)

    if not isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()

    return adata
