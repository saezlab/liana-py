import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.stats import norm


def _standardize_matrix(mat, local=True):
    mat = np.array(mat - np.array(mat.mean(axis=0)))
    if not local:
        mat = mat / np.sqrt(np.sum(mat ** 2, axis=0, keepdims=True))
    return mat


def _get_xy_matrices(x_mat, y_mat, x_pos, y_pos, x_order, y_order):
    x_mat = np.array([x_mat[:, x_pos[x]] for x in x_order])
    y_mat = np.array([y_mat[:, y_pos[y]] for y in y_order])

    assert x_mat.shape == y_mat.shape

    return x_mat, y_mat


def _global_permutation_pvals(x_mat, y_mat, dist, global_r, n_perm, positive_only, seed):
    assert isinstance(dist, csr_matrix)
    rng = np.random.default_rng(seed)

    # initialize mat /w n_perm * number of X->Y
    idx = x_mat.shape[1]

    # permutation mat /w n_perms x LR_n
    perm_mat = np.zeros((n_perm, global_r.shape[0]))

    for perm in tqdm(range(n_perm)):
        _idx = rng.permutation(idx)
        perm_mat[perm, :] = ((x_mat[:, _idx] @ dist) * y_mat).sum(axis=1)

    if positive_only:
        global_pvals = 1 - (global_r > perm_mat).sum(axis=0) / n_perm
    else:
        global_pvals = 2 * (1 - (abs(global_r) > abs(perm_mat)).sum(axis=0) / n_perm)

    return global_pvals


def _global_zscore_pvals(dist, global_r, positive_only):
    dist = np.array(dist.todense())
    spot_n = dist.shape[0]

    # global distance variance as in spatialDM
    numerator = spot_n ** 2 * ((dist * dist).sum()) - \
                (2 * spot_n * (dist @ dist).sum()) + \
                (dist.sum() ** 2)
    denominator = spot_n ** 2 * (spot_n - 1) ** 2
    dist_var_sq = (numerator / denominator) ** (1 / 2)

    global_zscores = global_r / dist_var_sq

    if positive_only:
        global_zpvals = norm.sf(global_zscores)
    else:
        global_zpvals = norm.sf(abs(global_zscores)) * 2

    return global_zpvals


def _local_permutation_pvals(x_mat, y_mat, local_r, dist, n_perm, seed, positive_only):
    rng = np.random.default_rng(seed)
    assert isinstance(dist, csr_matrix)

    spot_n = local_r.shape[1]  # n of 1:1 edges (e.g. lrs)
    xy_n = local_r.shape[0]

    # permutation cubes to be populated
    perm_x = np.zeros((xy_n, n_perm, spot_n))
    perm_y = np.zeros((xy_n, n_perm, spot_n))

    for i in tqdm(range(n_perm)):  # TODO fix spike in RAM
        _idx = rng.permutation(x_mat.shape[0])
        perm_x[:, i, :] = ((dist @ y_mat[_idx, :]) * x_mat).T
        perm_y[:, i, :] = ((dist @ x_mat[_idx, :]) * y_mat).T

    if positive_only:
        local_pvals = ((np.expand_dims(local_r, 1) <= (perm_x + perm_y)).sum(
            1)) / n_perm

        pos_msk = ((x_mat > 0) + (y_mat > 0)).T
        local_pvals[~pos_msk] = 1
    else:
        local_pvals = 1

    return local_pvals


def _local_zscore_pvals(x_mat, y_mat, local_r, dist, positive_only):
    spot_n = dist.shape[0]

    x_norm = np.array([norm.fit(x_mat[:, x]) for x in range(x_mat.shape[1])])
    y_norm = np.array([norm.fit(y_mat[:, y]) for y in range(y_mat.shape[1])])

    # get x,y std
    x_std, y_std = x_norm[:, 1], y_norm[:, 1]

    x_sigma = np.array([(std * spot_n / (spot_n - 1)) for std in x_std])
    y_sigma = np.array([(std * spot_n / (spot_n - 1)) for std in y_std])

    std = _get_local_var(x_sigma, y_sigma, dist, spot_n)
    local_zscores = local_r / std

    if positive_only:
        local_zpvals = norm.sf(local_zscores)
        pos_msk = ((x_mat > 0) + (y_mat > 0)).T
        local_zpvals[~pos_msk] = 1
    else:
        local_zpvals = norm.sf(abs(local_zscores))

    return local_zpvals


def _get_local_var(x_sigma, y_sigma, dist, spot_n):
    dist_sq = (np.array(dist.todense()) ** 2).sum(axis=1)

    n_weight = 2 * (spot_n - 1) ** 2 / spot_n ** 2
    sigma_prod = x_sigma * y_sigma
    core = n_weight * sigma_prod

    var = np.multiply.outer(dist_sq, core) + core
    std = var ** (1 / 2)

    return std.T
