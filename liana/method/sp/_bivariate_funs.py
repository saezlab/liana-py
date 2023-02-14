import numba as nb
import numpy as np
from scipy.stats import rankdata
from liana.method.sp._spatial_utils import _global_zscore_pvals, _global_permutation_pvals, _local_permutation_pvals, _local_zscore_pvals


@nb.njit(nb.float32(nb.float32[:], nb.float32[:], nb.float32[:], nb.float32, nb.boolean), cache=True)
def _wcorr(x, y, w, wsum, rank):
    
    if rank:
        x = np.argsort(x).argsort().astype(nb.float32)
        y = np.argsort(y).argsort().astype(nb.float32)
    
    wx = w * x
    wy = w * y
    
    numerator = wsum * sum(wx * y) - sum(wx) * sum(wy)
    
    denominator_x = wsum * sum(w * (x**2)) - sum(wx)**2
    denominator_y = wsum * sum(w * (y**2)) - sum(wy)**2
    denominator = (denominator_x * denominator_y)
    
    if (denominator == 0) or (numerator == 0):
        return 0
    
    return numerator / (denominator**0.5) ## TODO numba rounding issue?


@nb.njit(nb.float32(nb.float32[:], nb.float32[:], nb.float32[:], nb.float32, nb.int8), cache=True)
def _wcoex(x, y, w, wsum, method):
        if method == 0: # pearson
            c = _wcorr(x, y, w, wsum, False)
        elif method == 1: # spearman
            c = _wcorr(x, y, w, wsum, True)
            ## Any other method
        else: 
            raise ValueError("method not supported")
        return c



# 0 = pearson, 1 = spearman 
@nb.njit(nb.float32[:,:](nb.float32[:,:], nb.float32[:,:], nb.float32[:,:], nb.float32, nb.int8), parallel=True, cache=True)
def _masked_coexpressions(x_mat, y_mat, weight, weight_thr, method):
    spot_n = x_mat.shape[0]
    xy_n = x_mat.shape[1]
    
    local_correlations = np.zeros((spot_n, xy_n), dtype=nb.float32)
    
    for i in nb.prange(spot_n):
        w = weight[i, :]
        msk = w > weight_thr
        wsum = sum(w[msk])
        
        for j in range(xy_n):
            x = x_mat[:, j][msk]
            y = y_mat[:, j][msk]
            
            local_correlations[i, j] = _wcoex(x, y, w[msk], wsum, method)
    
    return local_correlations


@nb.njit(nb.float32[:,:](nb.float32[:,:], nb.float32[:,:], nb.float32[:,:], nb.float32), cache=True)
def _masked_pearson(x_mat, y_mat, weight, weight_thr):
    return _masked_coexpressions(x_mat, y_mat, weight, weight_thr, method=0)


@nb.njit(nb.float32[:,:](nb.float32[:,:], nb.float32[:,:], nb.float32[:,:], nb.float32), cache=True)
def _masked_spearman(x_mat, y_mat, weight, weight_thr):
    return _masked_coexpressions(x_mat, y_mat, weight, weight_thr, method=1)


def _vectorized_correlations(x_mat, y_mat, weight, method="pearson"):
    """
    Vectorized implementation of weighted correlations.
    
    Note: due to the imprecision of np.sum and np.dot, the function is accurate to 5 decimal places.
    
    """
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be one of 'pearson', 'spearman'")
    
    # transpose
    x_mat, y_mat = x_mat.T, y_mat.T
    

    weight_sums = np.sum(weight, axis = 0).flatten()
        
    if method=="spearman":
        x_mat = rankdata(x_mat, axis=1)
        y_mat = rankdata(y_mat, axis=1)
    
    # standard pearson
    n1 = (((x_mat * y_mat).dot(weight)) * weight_sums)
    n2 = (x_mat.dot(weight)) * (y_mat.dot(weight))
    numerator = n1 - n2
    
    denominator_x = (weight_sums * (x_mat ** 2).dot(weight)) - (x_mat.dot(weight))**2
    denominator_y = (weight_sums * (y_mat ** 2).dot(weight)) - (y_mat.dot(weight))**2
    denominator = (denominator_x * denominator_y)
    
    # numpy sum is unstable below 1e-6... 
    # results in the denominator being smaller than the numerator
    denominator[denominator < 1e-6] = 0
    denominator = denominator ** 0.5
    
    zeros = np.zeros(numerator.shape)
    local_corrs = np.divide(numerator, denominator, out=zeros, where=denominator!=0)
    
    # fix numpy imprecision, related to numba rounding issue? TODO check if it does not hide other issues
    local_corrs = np.clip(local_corrs, -1, 1, out=local_corrs)
    
    return local_corrs


def _vectorized_pearson(x_mat, y_mat, dist):
    return _vectorized_correlations(x_mat, y_mat, dist, method="pearson")


def _vectorized_spearman(x_mat, y_mat, dist):
    return _vectorized_correlations(x_mat, y_mat, dist, method="spearman")


def _vectorized_cosine(x_mat, y_mat, weight):
    x_mat, y_mat = x_mat.T, y_mat.T    
    
    xy_dot = (x_mat * y_mat).dot(weight)
    x_dot = (x_mat ** 2).dot(weight.T)
    y_dot = (y_mat ** 2).dot(weight.T)
    denominator = (x_dot * y_dot) + np.finfo(np.float32).eps
    
    return xy_dot / (denominator**0.5)


def _vectorized_jaccard(x_mat, y_mat, weight):
    # binarize
    x_mat, y_mat = x_mat > 0, y_mat > 0 ## TODO, only positive?
    # transpose
    x_mat, y_mat = x_mat.T, y_mat.T    
    
    # intersect and union
    numerator = np.dot(np.minimum(x_mat, y_mat), weight)
    denominator = np.dot(np.maximum(x_mat, y_mat), weight) + np.finfo(np.float32).eps
    
    return numerator / denominator


def _local_morans(x_mat, y_mat, dist):
    """

    Parameters
    ----------
    x_mat
        2D array with x variables
    y_mat
        2D array with y variables
    dist
    
    Returns
    -------
    Returns 2D array of local Moran's I with shape(n_spot, xy_n)

    """
    local_x = x_mat * (dist @ y_mat)
    local_y = x_mat * (dist @ y_mat)
    local_r = (local_x + local_y).T

    return local_r


def _handle_functions(function_name): # TODO improve this, maybe use a dict, or a class
    function_name = function_name.lower()
    
    if function_name == "pearson":
        return _vectorized_pearson
    elif function_name == "spearman":
        return _vectorized_spearman
    elif function_name == "masked_pearson":
        return _masked_pearson
    elif function_name == "masked_spearman":
        return _masked_spearman
    elif function_name == "cosine":
        return _vectorized_cosine
    elif function_name == "jaccard":
        return _vectorized_jaccard
    elif function_name == "morans":
        XXX
    else:
        raise ValueError("Function not implemented")



def _global_spatialdm(x_mat,
                      y_mat,
                      dist,
                      seed,
                      n_perm,
                      pvalue_method,
                      positive_only):
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
    dist
        proximity weight matrix, obtained e.g. via `liana.method.get_spatial_proximity`.
        Note that for spatialDM/Morans'I `dist` has to be weighed by n / sum(W).
    seed
        Reproducibility seed
    n_perm
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
    global_r = ((x_mat @ dist) * y_mat).sum(axis=1)

    # calc p-values
    if pvalue_method == 'permutation':
        global_pvals = _global_permutation_pvals(x_mat=x_mat,
                                                 y_mat=y_mat,
                                                 dist=dist,
                                                 global_r=global_r,
                                                 n_perm=n_perm,
                                                 positive_only=positive_only,
                                                 seed=seed
                                                 )
    elif pvalue_method == 'analytical':
        global_pvals = _global_zscore_pvals(dist=dist,
                                            global_r=global_r,
                                            positive_only=positive_only)

    return global_r, global_pvals




def _local_spatialdm(x_mat,
                     y_mat,
                     dist,
                     n_perm,
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
    dist
        proximity weight matrix, obtained e.g. via `liana.method.get_spatial_proximity`.
        Note that for spatialDM/Morans'I `dist` has to be weighed by n / sum(W).
    seed
        Reproducibility seed
    n_perm
        Number of permutatins to perform (if `pvalue_method`=='permutation')
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
    x_mat, y_mat = x_mat.T, y_mat.T
    local_r = _local_morans(x_mat, y_mat, dist)

    if pvalue_method == 'permutation':
        local_pvals = _local_permutation_pvals(x_mat=x_mat,
                                               y_mat=y_mat,
                                               dist=dist,
                                               local_truth=local_r,
                                               local_fun=_local_morans,
                                               n_perm=n_perm,
                                               seed=seed,
                                               positive_only=positive_only
                                               )
    elif pvalue_method == 'analytical':
        local_pvals = _local_zscore_pvals(x_mat=x_mat,
                                          y_mat=y_mat,
                                          local_r=local_r,
                                          dist=dist,
                                          positive_only=positive_only)

    return local_r.T, local_pvals.T


