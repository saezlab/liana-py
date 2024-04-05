from __future__ import annotations

import numpy as np
from pandas import Series

from scipy.sparse import csr_matrix, isspmatrix_csr, hstack
from scipy.stats import norm
from tqdm import tqdm
from anndata import AnnData
from liana._logging import _logg


class GlobalFunction:
    instances = {}

    def __init__(self, fun, name):
        self.fun = fun
        self.name = name
        self.pvals_name = self.name+ '_pvals'

        GlobalFunction.instances[name] = self

    def _permutation_pvals(self,
                           x_mat,
                           y_mat,
                           weight,
                           global_stat,
                           n_perms,
                           mask_negatives,
                           seed,
                           verbose
                           ):
        rng = np.random.default_rng(seed)

        # initialize mat /w n_perms * number of X->Y
        idx = x_mat.shape[0]

        # permutation mat /w n_permss x LR_n
        perm_mat = np.zeros((n_perms, global_stat.shape[0]))

        for perm in tqdm(range(n_perms), disable=not verbose):
            _idx = rng.permutation(idx)
            perm_mat[perm, :] = self.fun(x_mat=x_mat[_idx, :],
                                         y_mat=y_mat[_idx, :],
                                         weight=weight)

        if mask_negatives:
            global_pvals = 1 - (global_stat > perm_mat).sum(axis=0) / n_perms
        else:
            global_pvals = 1 - (np.abs(global_stat) > np.abs(perm_mat)).sum(axis=0) / n_perms

        return global_pvals


    def _zscore_pvals(self,
                      weight,
                      global_stat,
                      mask_negatives
                      ):
        """
        SpatialDM's global z-score p-value calculation

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

        global_zscores = global_stat / weight_var_sq

        if mask_negatives:
            global_zpvals = norm.sf(global_zscores)
        else:
            global_zpvals = norm.sf(np.abs(global_zscores)) * 2

        return global_zpvals


    def __call__(self,
                 xy_stats,
                 x_mat,
                 y_mat,
                 weight,
                 seed,
                 n_perms,
                 mask_negatives,
                 verbose
                 ):
        # NOTE: these are out of functions for permute efficiency
        if self.name == 'morans':
            x_mat = _zscore(x_mat, axis=0, global_r=True)
            y_mat = _zscore(y_mat, axis=0, global_r=True)
        if self.name == 'lee':
            x_mat = _zscore(x_mat)
            y_mat = _zscore(y_mat)
            weight = weight * weight

        global_stat = self.fun(x_mat=x_mat, y_mat=y_mat, weight=weight)


        if n_perms is None:
            global_pvals = None
        elif n_perms > 0:
            global_pvals = \
                self._permutation_pvals(x_mat=x_mat,
                                        y_mat=y_mat,
                                        weight=weight,
                                        global_stat=global_stat,
                                        n_perms=n_perms,
                                        mask_negatives=mask_negatives,
                                        seed=seed,
                                        verbose=verbose
                                        )
        elif n_perms==0 and self.name == 'morans':
            global_pvals = \
                self._zscore_pvals(weight=weight,
                                   global_stat=global_stat,
                                   mask_negatives=mask_negatives
                                   )
        elif n_perms==0 and self.name == 'lee':
            global_pvals = None
            _logg('Global Lee does not support analytical p-values', 'warning', verbose=verbose)

        xy_stats[self.name] = global_stat
        xy_stats[self.pvals_name] = global_pvals


def _global_r(x_mat, y_mat, weight):
    return ((weight @ x_mat) * y_mat).sum(axis=0)


def _global_l(x_mat, y_mat, weight):
    return ((weight @ x_mat) * y_mat).sum(axis=0) / weight.sum()

_global_r = GlobalFunction(_global_r, 'morans')
_global_l = GlobalFunction(_global_l, 'lee')


# NOTE: Common functions, move to utils
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


def _zscore(mat, axis=0, global_r=False):
    if global_r: # NOTE: specific to global SpatialDM
        spot_n = 1
    else:
        spot_n = mat.shape[axis]

    mat = mat - mat.mean(axis=axis)
    mat = mat / np.sqrt(np.sum(np.power(mat, 2), axis=axis) / spot_n)
    mat = np.clip(mat, -10, 10)

    return np.array(mat)


# TODO: Move to _bivariate_funs.py class
def _local_permutation_pvals(x_mat,
                             y_mat,
                             weight,
                             local_truth,
                             local_fun,
                             n_perms,
                             seed,
                             mask_negatives,
                             verbose):
    rng = np.random.default_rng(seed)

    spot_n = local_truth.shape[0]
    xy_n = local_truth.shape[1]

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

def _norm_max(X, axis=0):
    X = X / X.max(axis=axis).A
    X = _zscore(X, axis=axis)
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
