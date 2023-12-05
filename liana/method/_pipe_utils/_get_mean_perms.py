from __future__ import annotations

import anndata
import numpy as np
from tqdm import tqdm

from joblib import Parallel, delayed

def _get_means_perms(adata: anndata.AnnData,
                     n_perms: int,
                     seed: int,
                     agg_fun,
                     norm_factor: float | None,
                     n_jobs: int,
                     verbose: bool):
    """
    Generate permutations and indices required for permutation-based methods

    Parameters
    ----------
    adata
        Annotated data matrix
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

    if isinstance(norm_factor, np.float32):
        adata.X /= norm_factor

    # define labels and masks
    labels = adata.obs['@label'].cat.categories
    labels_mask = np.zeros((adata.shape[0], labels.shape[0]), dtype=bool)

    # populate masks shape(genes, labels)
    for ct_idx, label in enumerate(labels):
        labels_mask[:, ct_idx] = adata.obs['@label'] == label

    # Perm should be a cube /w dims: n_perms x idents x n_genes
    perms = _generate_perms_cube(adata.X, n_perms, labels_mask, seed, agg_fun, n_jobs, verbose)

    return perms


# Define a helper function for parallel processing
def _permute_and_aggregate(perm, perm_idx, X, labels_mask, agg_fun):
    perm_mat = X[perm_idx]
    permuted_means = np.array([agg_fun(perm_mat[labels_mask[:, i]], axis=0) for i in range(labels_mask.shape[1])])
    return perm, permuted_means


def _generate_perms_cube(X, n_perms, labels_mask, seed, agg_fun, n_jobs, verbose):
    # initialize rng
    rng = np.random.default_rng(seed=seed)

    # indexes to be shuffled
    idx = np.arange(X.shape[0])

    # Perm should be a cube /w dims: n_perms x idents x n_genes
    perms = np.zeros((n_perms, labels_mask.shape[1], X.shape[1]))

    # Use Parallel to enable parallelization
    results = Parallel(n_jobs=n_jobs)(delayed(_permute_and_aggregate)
                                      (perm, rng.permutation(idx), X, labels_mask, agg_fun)
                                      for perm in tqdm(range(n_perms), disable=not verbose)
                                      )

    # Unpack results
    for perm, permuted_means in results:
        perms[perm] = np.reshape(permuted_means, (labels_mask.shape[1], X.shape[1]))

    return perms


def _get_positions(adata, lr_res):
    labels = adata.obs['@label'].cat.categories

    # get positions of each entity in the matrix
    ligand_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity
                  in lr_res['ligand']}
    receptor_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity
                    in lr_res['receptor']}
    labels_pos = {labels[pos]: pos for pos in range(labels.shape[0])}

    return ligand_pos, receptor_pos, labels_pos


def _get_mat_idx(adata, lr_res):
    # convert to indexes
    ligand_pos, receptor_pos, labels_pos = _get_positions(adata, lr_res)

    ligand_idx = lr_res['ligand'].map(ligand_pos)
    receptor_idx = lr_res['receptor'].map(receptor_pos)

    source_idx = lr_res['source'].map(labels_pos)
    target_idx = lr_res['target'].map(labels_pos)

    return ligand_idx, receptor_idx, source_idx, target_idx



def _calculate_pvals(lr_truth, perm_stats, _score_fun):
    """
    Calculate p-values for a given DataFrame x and permutation statistics

    Parameters
    ----------
    x
        DataFrame with LIANA results
    perm_stats
        Permutation statistics (2 (ligand-receptor), n_perms (number of permutations, n_rows in lr_res)

    Returns
    -------
    A tuple with lr_mean and pvalue for x

    """
    # calculate p-values
    if perm_stats is not None:
        lr_perm_means = _score_fun(perm_stats, axis=0)
        n_perms = perm_stats.shape[1]
        pvals = np.sum(np.greater_equal(lr_perm_means, lr_truth), axis=0) / n_perms
    else:
        pvals = None


    return pvals
