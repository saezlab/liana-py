from __future__ import annotations

import numpy as np
from pandas import DataFrame, Series

from scipy.sparse import csr_matrix, isspmatrix_csr, hstack
from scipy.stats import norm
from tqdm import tqdm
from anndata import AnnData


def _rename_means(lr_stats, entity):
    df = lr_stats.copy()
    df.columns = df.columns.map(lambda x: entity + '_' + str(x) if x != 'gene' else 'gene')
    return df.rename(columns={'gene': entity})


def _local_to_dataframe(array, idx, columns):
    if array is None:
        return None
    return DataFrame(array.T, index=idx, columns=columns)


def _local_permutation_pvals(x_mat, y_mat, weight, local_truth, local_fun, n_perms, seed,
                             positive_only, pos_msk, verbose):
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
    for i in tqdm(range(n_perms), disable=not verbose):
        _idx = rng.permutation(spot_n)
        perm_score = local_fun(x_mat=x_mat[_idx, :], y_mat=y_mat, weight=weight)
        if positive_only:
            local_pvals += np.array(perm_score >= local_truth, dtype=int)
        else:
            local_pvals += (np.array(np.abs(perm_score) >= np.abs(local_truth), dtype=int))

    local_pvals = local_pvals / n_perms

    if positive_only:
        local_pvals[~pos_msk] = 1

    return local_pvals


def _zscore(mat, local=True, axis=0):
    spot_n = mat.shape[1]
    
    # NOTE: specific to global SpatialDM
    if not local:
        spot_n = 1
    
    mat = np.array(mat - np.array(mat.mean(axis=axis)))
    mat = mat / np.sqrt(np.sum(mat ** 2, axis=axis, keepdims=True) / spot_n)
    
    return mat


def _encode_as_char(a, weight):
    # if only positive
    if np.all(a >= 0):
        a = _zscore(a.T).T
    
    # to get a sign for each spot, we multiply by connectivities
    a = a @ weight
    
    a = np.where(a > 0, 'P', np.where(a < 0, 'N', 'Z'))
    return a


def _categorize(x_mat, y_mat, weight, idx, columns):
    x_cats = _encode_as_char(x_mat.A, weight)
    y_cats = _encode_as_char(y_mat.A, weight)
    
    # add the two categories, and simplify them to ints
    cats = np.core.defchararray.add(x_cats, y_cats)
    cats = _simplify_cats(cats)
    
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
    

def _global_permutation_pvals(x_mat, y_mat, weight, global_r, n_perms, positive_only, seed, verbose):
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

    for perm in tqdm(range(n_perms), disable=not verbose):
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
        connectivity weight matrix (spot_n x spot_n)
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


def _local_zscore_pvals(x_mat, y_mat, local_truth, weight, positive_only, pos_msk):
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
        connectivity weight matrix
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
                      positive_only,
                      verbose
                      ):
    # Get global r
    global_r = ((x_mat @ weight) * y_mat).sum(axis=1)

    # calc p-values
    
    if n_perms is None:
        global_pvals = None
    elif n_perms > 0:
        global_pvals = _global_permutation_pvals(x_mat=x_mat,
                                                 y_mat=y_mat,
                                                 weight=weight,
                                                 global_r=global_r,
                                                 n_perms=n_perms,
                                                 positive_only=positive_only,
                                                 seed=seed,
                                                 verbose=verbose
                                                 )
    elif n_perms==0:
        global_pvals = _global_zscore_pvals(weight=weight,
                                            global_r=global_r,
                                            positive_only=positive_only)

    return np.array((global_r, global_pvals))


def _run_scores_pipeline(xy_stats, x_mat, y_mat, idx, local_fun, pos_msk,
                         weight, positive_only, n_perms, seed, verbose):
    local_scores, local_pvals = _get_local_scores(x_mat=x_mat.T,
                                                  y_mat=y_mat.T,
                                                  local_fun=local_fun,
                                                  weight=weight,
                                                  seed=seed,
                                                  n_perms=n_perms,
                                                  positive_only=positive_only,
                                                  pos_msk=pos_msk,
                                                  verbose=verbose
                                                  )

    # global scores fun
    xy_stats = _get_global_scores(xy_stats=xy_stats,
                                  x_mat=x_mat,
                                  y_mat=y_mat,
                                  local_fun=local_fun,
                                  weight=weight,
                                  seed=seed,
                                  n_perms=n_perms,
                                  positive_only=positive_only,
                                  local_scores=local_scores,
                                  verbose=verbose
                                  )

    # convert to DataFrames
    local_scores = _local_to_dataframe(array=local_scores,
                                       idx=idx,
                                       columns=xy_stats['interaction'])

    return xy_stats, local_scores, local_pvals


def _get_local_scores(x_mat,
                      y_mat,
                      local_fun,
                      weight,
                      n_perms,
                      seed,
                      positive_only,
                      pos_msk,
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
        x_mat = _zscore(x_mat, local=True, axis=0)
        y_mat = _zscore(y_mat, local=True, axis=0)
        
        # # NOTE: spatialdm do this, and also use .raw by default
        # x_mat = x_mat / np.max(x_mat, axis=0)
        # y_mat = y_mat / np.max(y_mat, axis=0)
        
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
                                               positive_only=positive_only,
                                               pos_msk=pos_msk,
                                               verbose=verbose
                                               )
    elif n_perms == 0:
        local_pvals = _local_zscore_pvals(x_mat=x_mat,
                                          y_mat=y_mat,
                                          local_truth=local_scores,
                                          weight=weight,
                                          positive_only=positive_only,
                                          pos_msk=pos_msk
                                          )

    return local_scores, local_pvals


def _get_global_scores(xy_stats, x_mat, y_mat, local_fun, weight, positive_only,
                       n_perms, seed, local_scores, verbose):
    if local_fun.__name__ == "_local_morans":
        xy_stats.loc[:, ['global_r', 'global_pvals']] = \
            _global_spatialdm(x_mat=_zscore(x_mat, local=False, axis=1),
                              y_mat=_zscore(y_mat, local=False, axis=1),
                              weight=weight,
                              seed=seed,
                              n_perms=n_perms,
                              positive_only=positive_only,
                              verbose=verbose
                              ).T
    else:
        # any other local score
        xy_stats.loc[:, ['global_mean', 'global_sd']] = np.vstack(
            [np.mean(local_scores, axis=1), np.std(local_scores, axis=1)]
            ).T

    return xy_stats


def _connectivity_to_weight(connectivity, local_fun):
    connectivity = csr_matrix(connectivity, dtype=np.float32)
    
    if local_fun.__name__ == "_local_morans":
        norm_factor = connectivity.shape[0] / connectivity.sum()
        connectivity = norm_factor * connectivity
        
        return csr_matrix(connectivity)
    
    # NOTE vectorized is faster with non-sparse, masked_scores won't work with sparse
    elif (connectivity.shape[0] < 5000) | local_fun.__name__.__contains__("masked"):
            return connectivity.A
    else:
        return csr_matrix(connectivity)


def _handle_connectivity(adata, connectivity, connectivity_key):
    if connectivity is None:
        if adata.obsp[connectivity_key] is None:
            raise ValueError(f'No connectivity matrix founds in mdata.obsp[{connectivity_key}]')
        connectivity = adata.obsp[connectivity_key]
        
    if not isspmatrix_csr(connectivity):
        connectivity = csr_matrix(connectivity, dtype=np.float32)
        
    return connectivity


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
