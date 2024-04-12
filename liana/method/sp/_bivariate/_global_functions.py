from __future__ import annotations

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from liana._logging import _logg
from liana.method.sp._utils import _zscore, _spatialdm_weight_norm

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
        if self.name == 'morans':
            x_mat = _zscore(x_mat, axis=0, global_r=True)
            y_mat = _zscore(y_mat, axis=0, global_r=True)
            weight = _spatialdm_weight_norm(weight)
        elif self.name == 'lee':
            x_mat = _zscore(x_mat)
            y_mat = _zscore(y_mat)
            weight = weight * weight
        else:
            raise ValueError('Global function not supported')

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
