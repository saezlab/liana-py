import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.stats import norm

from anndata import AnnData
from pandas import DataFrame
from typing import Optional

from liana.method.sp._SpatialMethod import SpatialMethod
from ._global_lr_pipe import _global_lr_pipe


class SpatialDM(SpatialMethod):
    def __init__(self, _method, _complex_cols, _obsm_keys):
        super().__init__(method_name=_method.method_name,
                         key_cols=_method.key_cols,
                         reference=_method.reference,
                         )

        self.complex_cols = _complex_cols
        self.obsm_keys = _obsm_keys
        self._method = _method

    def __call__(self,
                 adata: AnnData,
                 resource_name: str = 'consensus',
                 expr_prop: float = 0.05,
                 pvalue_method: str = 'permutation',
                 n_perm=1000,
                 positive_only: bool = True,
                 use_raw: Optional[bool] = True,
                 layer: Optional[str] = None,
                 verbose: Optional[bool] = False,
                 seed: int = 1337,
                 resource: Optional[DataFrame] = None,
                 inplace=True):
        """
        Parameters
        ----------
        adata
            Annotated data object.
        resource_name
            Name of the resource to be loaded and use for ligand-receptor inference.
        expr_prop
            Minimum expression proportion for the ligands/receptors (and their subunits).
             Set to `0` to return unfiltered results.
        pvalue_method
            Method to obtain P-values: ['permutation', 'analytical'].
        n_perm
            Number of permutations to be performed if `pvalue_method`=='permutation'
        positive_only
            Whether to calculate p-values only for positive correlations. `True` by default.
        use_raw
            Use raw attribute of adata if present.
        layer
            Layer in anndata.AnnData.layers to use. If None, use anndata.AnnData.X.
        verbose
            Verbosity flag
        seed
            Random seed for reproducibility.
        resource
            Parameter to enable external resources to be passed. Expects a pandas dataframe
            with [`ligand`, `receptor`] columns. None by default. If provided will overrule
            the resource requested via `resource_name`
        inplace
            If true return `DataFrame` with results, else assign to `.uns`.

        Returns
        -------

        """
        assert pvalue_method in ['analytical', 'permutation']

        temp, lr_res, ligand_pos, receptor_pos = _global_lr_pipe(adata=adata,
                                                                 resource_name=resource_name,
                                                                 resource=resource,
                                                                 expr_prop=expr_prop,
                                                                 use_raw=use_raw,
                                                                 layer=layer,
                                                                 verbose=verbose,
                                                                 _key_cols=self.key_cols,
                                                                 _complex_cols=self.complex_cols,
                                                                 _obms_keys=self.obsm_keys
                                                                 )

        # n / sum(W) for Moran's I
        norm_factor = temp.obsm['proximity'].shape[0] / temp.obsm['proximity'].sum()
        dist = csr_matrix(norm_factor * temp.obsm['proximity'])

        lr_res['global_r'], lr_res['global_pvals'] = \
            _global_spatialdm(mat=temp.X,
                              ligand_pos=ligand_pos,
                              receptor_pos=receptor_pos,
                              lr_res=lr_res,
                              dist=dist,
                              seed=seed,
                              n_perm=n_perm,
                              pvalue_method=pvalue_method,
                              positive_only=positive_only
                              )
        adata.uns['global_res'] = lr_res

        adata.obsm['local_r'], adata.obsm['local_pvals'] = \
            _local_spatialdm(mat=temp.X,
                             ligand_pos=ligand_pos,
                             receptor_pos=receptor_pos,
                             lr_res=lr_res,
                             dist=dist,
                             seed=seed,
                             n_perm=n_perm,
                             pvalue_method=pvalue_method,
                             positive_only=positive_only
                             )

        # convert to dataframes
        adata.obsm['local_r'] = _local_to_dataframe(array=adata.obsm['local_r'],
                                                    idx=temp.obs.index,
                                                    columns=lr_res.interaction)
        adata.obsm['local_pvals'] = _local_to_dataframe(array=adata.obsm['local_pvals'],
                                                        idx=temp.obs.index,
                                                        columns=lr_res.interaction)

        return None if inplace else lr_res


def _global_spatialdm(mat,
                      ligand_pos,
                      receptor_pos,
                      lr_res,
                      dist,
                      seed,
                      n_perm,
                      pvalue_method,
                      positive_only):
    # normalize matrix
    mat = _standardize_matrix(mat, local=False)

    # spot_n x lr_n matrices
    ligand_mat, receptor_mat = _get_xy_matrices(x_mat=mat,
                                                y_mat=mat,
                                                x_pos=ligand_pos,
                                                y_pos=receptor_pos,
                                                x_order=lr_res.ligand,
                                                y_order=lr_res.receptor)

    # Get global r
    global_r = ((ligand_mat @ dist) * receptor_mat).sum(axis=1)

    # calc p-values
    if pvalue_method == 'permutation':
        global_pvals = _global_permutation_pvals(x_mat=ligand_mat,
                                                 y_mat=receptor_mat,
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


def _local_spatialdm(mat,
                     ligand_pos,
                     receptor_pos,
                     lr_res,
                     dist,
                     n_perm,
                     seed,
                     pvalue_method,
                     positive_only
                     ):
    mat = _standardize_matrix(mat, local=True)
    ligand_mat, receptor_mat = _get_xy_matrices(x_mat=mat, y_mat=mat,
                                                x_pos=ligand_pos, y_pos=receptor_pos,
                                                x_order=lr_res.ligand, y_order=lr_res.receptor)
    ligand_mat, receptor_mat = ligand_mat.T, receptor_mat.T
    # calculate local_Rs
    local_x = ligand_mat * (dist @ receptor_mat)
    local_y = receptor_mat * (dist @ ligand_mat)
    local_r = (local_x + local_y).T

    if pvalue_method == 'permutation':
        local_pvals = _local_permutation_pvals(x_mat=ligand_mat, y_mat=receptor_mat,
                                               dist=dist, local_r=local_r,
                                               n_perm=n_perm, seed=seed,
                                               positive_only=positive_only
                                               )
    elif pvalue_method == 'analytical':
        local_pvals = _local_zscore_pvals(x_mat=ligand_mat, y_mat=receptor_mat,
                                          local_r=local_r, dist=dist,
                                          positive_only=positive_only)

    return local_r.T, local_pvals.T


def _local_to_dataframe(idx, columns, array):
    return DataFrame(array, index=idx, columns=columns)


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
        # TODO Proof this makes sense
        global_pvals = 2 * (1 - (np.abs(global_r) > np.abs(perm_mat)).sum(axis=0) / n_perm)

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
        global_zpvals = norm.sf(np.abs(global_zscores)) * 2

    return global_zpvals


def _local_permutation_pvals(x_mat, y_mat, local_r, dist, n_perm, seed, positive_only):
    rng = np.random.default_rng(seed)
    assert isinstance(dist, csr_matrix)

    spot_n = local_r.shape[1]  # n of 1:1 edges (e.g. lrs)
    xy_n = local_r.shape[0]

    # permutation cubes to be populated
    local_pvals = np.zeros((xy_n, spot_n))

    for i in tqdm(range(n_perm)):
        _idx = rng.permutation(x_mat.shape[0])
        perm_x = ((dist @ y_mat[_idx, :]) * x_mat).T
        perm_y = ((dist @ x_mat[_idx, :]) * y_mat).T
        perm_r = perm_x + perm_y
        if positive_only:
            local_pvals += np.array(perm_r >= local_r, dtype=int)
        else:
            # TODO Proof this makes sense
            local_pvals += np.array(np.abs(perm_r) >= np.abs(local_r), dtype=int)

    local_pvals = local_pvals / n_perm

    if positive_only:  # mask?
        # only keep positive pvals where either x or y is positive
        pos_msk = ((x_mat > 0) + (y_mat > 0)).T
        local_pvals[~pos_msk] = 1

    return local_pvals


def _local_zscore_pvals(x_mat, y_mat, local_r, dist, positive_only):
    spot_n = dist.shape[0]

    x_norm = np.array([norm.fit(x_mat[:, x]) for x in range(x_mat.shape[1])])  # TODO to np.repeat
    y_norm = np.array([norm.fit(y_mat[:, y]) for y in range(y_mat.shape[1])])

    # get x,y std
    x_std, y_std = x_norm[:, 1], y_norm[:, 1]

    x_sigma = np.array([(std * spot_n / (spot_n - 1)) for std in x_std])  # TODO to np.repeat
    y_sigma = np.array([(std * spot_n / (spot_n - 1)) for std in y_std])

    std = _get_local_var(x_sigma, y_sigma, dist, spot_n)
    local_zscores = local_r / std

    if positive_only:
        local_zpvals = norm.sf(local_zscores)
        pos_msk = ((x_mat > 0) + (y_mat > 0)).T  # mask?
        local_zpvals[~pos_msk] = 1
    else:
        local_zpvals = norm.sf(np.abs(local_zscores))

    return local_zpvals


def _get_local_var(x_sigma, y_sigma, dist, spot_n):
    dist_sq = (np.array(dist.todense()) ** 2).sum(axis=1)

    n_weight = 2 * (spot_n - 1) ** 2 / spot_n ** 2
    sigma_prod = x_sigma * y_sigma
    core = n_weight * sigma_prod

    var = np.multiply.outer(dist_sq, core) + core
    std = var ** (1 / 2)

    return std.T


# initialize instance
_spatialdm = SpatialMethod(
    method_name="SpatialDM",
    key_cols=['ligand_complex', 'receptor_complex'],
    reference="Zhuoxuan, L.I., Wang, T., Liu, P. and Huang, Y., 2022. SpatialDM: Rapid "
              "identification of spatially co-expressed ligand-receptor reveals cell-cell "
              "communication patterns. bioRxiv. "
)

spatialdm = SpatialDM(_method=_spatialdm,
                      _complex_cols=['ligand_means', 'receptor_means'],
                      _obsm_keys=['proximity']
                      )
