import numba as nb
import numpy as np
from scipy.stats import rankdata,norm
from scipy.sparse import issparse
from tqdm import tqdm
from liana.method.sp._utils import _zscore, _spatialdm_weight_norm


class LocalFunction:
    """
    Class representing information about bivariate spatial functions.
    """

    instances = {}

    def __init__(self, name, metadata, fun, reference=None):
        self.name = name
        self.metadata = metadata
        self.fun = fun
        self.reference = reference

        LocalFunction.instances[name] = self

    def __call__(self,
                 x_mat,
                 y_mat,
                 weight,
                 n_perms,
                 seed,
                 mask_negatives,
                 verbose
                 ):
        if self.name == 'morans':
            x_mat = self._norm_max(x_mat)
            y_mat = self._norm_max(y_mat)
            weight = _spatialdm_weight_norm(weight)
        else:
            if issparse(x_mat):
               x_mat = x_mat.A
            if issparse(y_mat):
                y_mat = y_mat.A

        if self.name.__contains__("masked") or weight.shape[0] < 10000:
            weight = weight.todense().A

        local_scores = self.fun(x_mat, y_mat, weight)

        if n_perms is None:
            local_pvals = None
        elif n_perms > 0:
            local_pvals = self._permutation_pvals(x_mat=x_mat,
                                                  y_mat=y_mat,
                                                  weight=weight,
                                                  local_truth=local_scores,
                                                  n_perms=n_perms,
                                                  seed=seed,
                                                  mask_negatives=mask_negatives,
                                                  verbose=verbose
                                                  )
        elif n_perms == 0:
            local_pvals = self._zscore_pvals(x_mat=x_mat,
                                             y_mat=y_mat,
                                             weight=weight,
                                             local_truth=local_scores,
                                             mask_negatives=mask_negatives
                                             )

        return local_scores, local_pvals

    def __repr__(self):
        return f"{self.name}: {self.metadata}"

    def _permutation_pvals(self,
                           x_mat,
                           y_mat,
                           weight,
                           local_truth,
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
            perm_score = self.fun(x_mat=x_mat[_idx, :], y_mat=y_mat[_idx, :], weight=weight)
            if mask_negatives:
                local_pvals += np.array(perm_score >= local_truth, dtype=int)
            else:
                local_pvals += np.array(np.abs(perm_score) >= np.abs(local_truth), dtype=int)

        local_pvals = local_pvals / n_perms

        return local_pvals


    def _zscore_pvals(self, x_mat, y_mat, local_truth, weight, mask_negatives):
        """
        Local Moran's R analytical p-values as in spatialDM (Li et al., 2022)


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

        std = self._get_local_var(x_sigma, y_sigma, weight, spot_n)
        local_zscores = local_truth / std

        if mask_negatives:
            local_zpvals = norm.sf(local_zscores)
        else:
            local_zpvals = norm.sf(np.abs(local_zscores))

        return local_zpvals


    def _get_local_var(self, x_sigma, y_sigma, weight, spot_n):
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

    def _norm_max(self, X, axis=0):
        X = X / X.max(axis=axis).A
        X = _zscore(X, axis=axis)
        X = np.where(np.isnan(X), 0, X)

        return X

    @classmethod
    def _get_instance(cls, name):
        name = name.lower()
        instances = cls.instances
        if name not in instances:
            raise ValueError(f"Function {name} not found. Available functions are: {', '.join(instances)}")

        return cls.instances[name]


@nb.njit(nb.float32(nb.float32[:], nb.float32[:], nb.float32[:], nb.float32), cache=True)
def _wcorr(x, y, w, wsum):

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

    return numerator / (denominator**0.5)


@nb.njit(nb.float32[:,:](nb.float32[:,:], nb.float32[:,:], nb.float32[:,:]), parallel=True, cache=True)
def _masked_spearman(x_mat, y_mat, weight):
    spot_n = x_mat.shape[0]
    xy_n = x_mat.shape[1]

    local_corrs = np.zeros((spot_n, xy_n), dtype=nb.float32)

    for i in nb.prange(spot_n):
        w = weight[i, :]
        msk = w > 0
        wsum = sum(w[msk])

        for j in range(xy_n):
            x = x_mat[:, j][msk]
            y = y_mat[:, j][msk]

            local_corrs[i, j] = _wcorr(x, y, w[msk], wsum)

    # NOTE done due to numpy/numba sum imprecision, https://github.com/numba/numba/issues/8749
    local_corrs = np.clip(a=local_corrs, a_min=-1.0, a_max=1.0, out=local_corrs)

    return local_corrs


def _vectorized_correlations(x_mat, y_mat, weight, method="pearson"):
    """
    Vectorized implementation of weighted correlations.

    Note: due to the imprecision of np.sum and np.dot, the function is accurate to 5 decimal places.

    """
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be one of 'pearson', 'spearman'")
    weight_sums = np.array(np.sum(weight, axis=1)).reshape(-1, 1)

    if method=="spearman":
        x_mat = rankdata(x_mat, axis=0)
        y_mat = rankdata(y_mat, axis=0)

    # standard pearson
    n1 = weight_sums * (weight @ (x_mat * y_mat))
    n2 = (weight @ x_mat) * (weight @ y_mat)
    numerator = n1 - n2

    denominator_x = (weight_sums * (weight @ x_mat ** 2)) - (weight @ x_mat)**2
    denominator_y = (weight_sums * (weight @ y_mat ** 2)) - (weight @ y_mat)**2
    denominator = denominator_x * denominator_y

    # numpy sum is unstable below 1e-6...
    denominator[denominator < 1e-6] = 0
    denominator = denominator ** 0.5

    zeros = np.zeros(numerator.shape)
    local_corrs = np.divide(numerator, denominator, out=zeros, where=denominator!=0)

    # NOTE done due to numpy/numba sum imprecision, https://github.com/numba/numba/issues/8749
    local_corrs = np.clip(local_corrs, -1, 1, out=local_corrs, dtype=np.float32)

    return local_corrs


def _vectorized_pearson(x_mat, y_mat, weight):
    return _vectorized_correlations(x_mat, y_mat, weight, method="pearson")


def _vectorized_spearman(x_mat, y_mat, weight):
    return _vectorized_correlations(x_mat, y_mat, weight, method="spearman")


def _vectorized_cosine(x_mat, y_mat, weight):
    xy_dot = weight @ (x_mat * y_mat)
    x_dot = weight @ (x_mat**2)
    y_dot = weight @ (y_mat**2)
    denominator = (x_dot * y_dot) + np.finfo(np.float32).eps

    return xy_dot / denominator**0.5


def _vectorized_jaccard(x_mat, y_mat, weight):
    x_mat, y_mat = x_mat > 0, y_mat > 0 ## NOTE only positive
    numerator = weight @ np.minimum(x_mat, y_mat)
    denominator = weight @ np.maximum(x_mat, y_mat) + np.finfo(np.float32).eps

    return numerator / denominator


def _local_morans(x_mat, y_mat, weight):
    """

    Parameters
    ----------
    x_mat
        2D array with x variables
    y_mat
        2D array with y variables

    Returns
    -------
    Returns 2D array of local Moran's I with shape(n_spot, xy_n)

    """
    local_x = x_mat * (weight @ y_mat)
    local_y = y_mat * (weight @ x_mat)
    local_r = (local_x + local_y)

    return local_r


def _product(x_mat, y_mat, weight):
    x_mat = weight @ x_mat
    y_mat = weight @ y_mat
    score = x_mat * y_mat

    return score


def _norm_product(x_mat, y_mat, weight):
    x_mat = weight @ x_mat
    y_mat = weight @ y_mat

    x_norm = np.max(np.abs(x_mat), axis=0)
    y_norm = np.max(np.abs(y_mat), axis=0)

    x_norm[x_norm == 0.] = 1.
    y_norm[y_norm == 0.] = 1.

    x_mat = x_mat / x_norm
    y_mat = y_mat / y_norm

    score = x_mat * y_mat

    return score

_bivariate_functions = [
        LocalFunction(
            name="pearson",
            metadata="weighted Pearson correlation coefficient",
            fun = _vectorized_pearson,
        ),
        LocalFunction(
            name="spearman",
            metadata="weighted Spearman correlation coefficient",
            fun = _vectorized_spearman,
        ),
        LocalFunction(
            name="cosine",
            metadata="weighted Cosine similarity",
            fun = _vectorized_cosine,
        ),
        LocalFunction(
            name="jaccard",
            metadata="weighted Jaccard similarity",
            fun = _vectorized_jaccard,
        ),
        LocalFunction(
            name="product",
            metadata="simple weighted product",
            fun = _product,
            reference="If vars are z-scaled = Lee's static (Lee 2021;J.Geograph.Syst.)"
        ),
        LocalFunction(
            name="norm_product",
            metadata="normalized weighted product",
            fun = _norm_product,
        ),
        LocalFunction(
            name="morans",
            metadata="Moran's R",
            fun=_local_morans,
            reference="Li, Z., Wang, T., Liu, P. and Huang, Y., 2022. SpatialDM:"
            "Rapid identification of spatially co-expressed ligand-receptor"
            "reveals cell-cell communication patterns. bioRxiv, pp.2022-08."
        ),
        LocalFunction(
            name= "masked_spearman",
            metadata="masked & weighted Spearman correlation",
            fun=_masked_spearman,
            reference="Ghazanfar, S., Lin, Y., Su, X., Lin, D.M., Patrick, E., Han, Z.G., Marioni, J.C. and Yang, J.Y.H., 2020."
            "Investigating higher-order interactions in single-cell data with scHOT. Nature methods, 17(8), pp.799-806."
        ),
    ]
