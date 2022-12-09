import anndata
import numpy as np
import pandas
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF


def _get_means_perms(adata: anndata.AnnData,
                     lr_res: pandas.DataFrame,
                     n_perms: int,
                     seed: int,
                     agg_fun,
                     norm_factor: (float, None),
                     verbose: bool):
    """
    Generate permutations and indices required for permutation-based methods

    Parameters
    ----------
    adata
        Annotated data matrix.
    lr_res
        Ligand-receptor stats DataFrame
    n_perms
        Number of permutations to be calculated
    seed
        Random seed for reproducibility.
    agg_fun
        function by which to aggregate the matrix, should take `axis` argument
    norm_factor
        additionally normalize the data by some factor (e.g. matrix max for CellChat)
    verbose
        Verbosity bool

    Returns
    -------
    Tuple with:
        - perms: 3D tensor with permuted averages per cluster
        - ligand_pos: Index of the ligand in the tensor
        - receptor_pos: Index of the receptor in the perms tensor
        - labels_pos: Index of cell identities in the perms tensor
    """

    # initialize rng
    rng = np.random.default_rng(seed=seed)

    if isinstance(norm_factor, np.float):
        adata.X /= norm_factor

    # define labels and dict
    labels = adata.obs.label.cat.categories
    labels_dict = {label: adata.obs.label.isin([label]) for label in labels}

    # indexes to be shuffled
    idx = np.arange(adata.X.shape[0])

    # Perm should be a cube /w dims: n_perms x idents x n_genes
    perms = np.zeros((n_perms, labels.shape[0], adata.shape[1]))

    # Assign permuted matrix
    for perm in tqdm(range(n_perms), disable=not verbose):
        perm_idx = rng.permutation(idx)
        perm_mat = adata.X[perm_idx]
        # populate matrix /w permuted means
        for cind in range(labels.shape[0]):
            perms[perm, cind] = agg_fun(perm_mat[labels_dict[labels[cind]]], axis=0)

    # Get indexes for each gene and label in the permutations
    ligand_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity
                  in lr_res['ligand']}
    receptor_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity
                    in lr_res['receptor']}
    labels_pos = {labels[pos]: pos for pos in range(labels.shape[0])}

    return perms, ligand_pos, receptor_pos, labels_pos


def _get_lr_pvals(x, perms, ligand_pos, receptor_pos, labels_pos, agg_fun,
                  ligand_col='ligand_means', receptor_col='receptor_means'):
    """
    Calculate Permutation means and p-values

    Parameters
    ----------
    x
        DataFrame row
    perms
        3D tensor with permuted averages per cluster
    ligand_pos
        Index of the ligand in the tensor
    receptor_pos
        Index of the receptor in the perms tensor
    labels_pos
        Index of cell identities in the perms tensor
    agg_fun
        function to aggregate the ligand and receptor

    Returns
    -------
    A tuple with lr_score (aggregated according to `agg_fun`) and ECDF p-value for x

    """
    # actual lr_score
    lr_score = agg_fun(x[ligand_col], x[receptor_col])

    if lr_score == 0:
        return 0, 1

    # Permutations lr mean
    ligand_perm_means = perms[:, labels_pos[x.source], ligand_pos[x.ligand]]
    receptor_perm_means = perms[:, labels_pos[x.target], receptor_pos[x.receptor]]
    lr_perm_score = agg_fun(ligand_perm_means, receptor_perm_means)

    p_value = (1 - ECDF(lr_perm_score)(lr_score))

    return lr_score, p_value
