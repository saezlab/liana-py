from __future__ import annotations

import anndata
import numpy as np
from tqdm import tqdm


def _get_means_perms(adata: anndata.AnnData,
                     n_perms: int,
                     seed: int,
                     agg_fun,
                     norm_factor: float | None,
                     verbose: bool,
                     met: bool = False,):
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

    if isinstance(norm_factor, np.float):
        adata.X /= norm_factor

    # define labels and masks
    labels = adata.obs.label.cat.categories
    labels_mask = np.zeros((adata.shape[0], labels.shape[0]), dtype=bool)
    
    # populate masks shape(genes, labels)
    for ct_idx, label in enumerate(labels):
        labels_mask[:, ct_idx] = adata.obs.label == label

    # generate two cube for metabolites and genes
    if met:
        perms = _generate_perms_cube(adata.X, n_perms, labels_mask, seed, agg_fun, verbose, met=met, Y=adata.obsm['metabolite_abundance'])
        perms_ligand = perms[0]
        perms_receptor = perms[1]

        return perms_ligand, perms_receptor
    
    else:
        # Perm should be a cube /w dims: n_perms x idents x n_genes
        perms = _generate_perms_cube(adata.X, n_perms, labels_mask, seed, agg_fun, verbose)

        return perms


# @nb.njit(nb.float32[:,:,:](nb.float32[:,:], nb.int32[:,:], nb.float32[:]), cache=True)
def _generate_perms_cube(X, n_perms, labels_mask, seed, agg_fun, verbose, met=False, Y=None):
    # initialize rng
    rng = np.random.default_rng(seed=seed)
    
    # indexes to be shuffled
    idx = np.arange(X.shape[0])

    if met:
        perms_receptors = np.zeros((n_perms, labels_mask.shape[0], X.shape[1]))
        perms_ligands = np.zeros((n_perms, labels_mask.shape[0], Y.shape[1]))
    else:
        # Perm should be a cube /w dims: n_perms x idents x n_genes
        perms = np.zeros((n_perms, labels_mask.shape[1], X.shape[1])) # , dtype=np.float32

    # Assign permuted matrix
    for perm in tqdm(range(n_perms), disable=not verbose):
        perm_idx = rng.permutation(idx)
        perm_mat = X[perm_idx]
        if met:
            perm_mat2 = Y[perm_idx]
            # perms_receptors[perm] = np.squeeze(np.array([agg_fun(perm_mat[labels_mask[label]], axis=0) for label in labels_mask]))
            # perms_ligands[perm] = np.squeeze(np.array([agg_fun(perm_mat2[labels_mask[label]], axis=0) for label in labels_mask]))

            # return perms_ligands, perms_receptors
        # populate matrix /w permuted means
        for ct_idx in range(labels_mask.shape[1]):
            ct_mask = labels_mask[:, ct_idx]
            if met:
                perms_receptors[perm, ct_idx] = agg_fun(perm_mat[ct_mask], axis=0)
                perms_ligands[perm, ct_idx] = agg_fun(perm_mat2[ct_mask], axis=0)

                return perms_ligands, perms_receptors
            else:
                perms[perm, ct_idx] = agg_fun(perm_mat[ct_mask], axis=0)
                   
                return perms


def _get_positions(adata, lr_res, met = False):
    labels = adata.obs['label'].cat.categories
    
    # get positions of each entity in the matrix
    if met:
        ligand_pos = {entity: np.where(adata.uns['met_index'] == entity)[0][0] for \
                      entity in lr_res['ligand']}
    else:
        ligand_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity \
                  in lr_res['ligand']}
    receptor_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity \
                    in lr_res['receptor']}
    labels_pos = {labels[pos]: pos for pos in range(labels.shape[0])}
    
    return ligand_pos, receptor_pos, labels_pos


def _get_mat_idx(adata, lr_res, met = False):
    # convert to indexes
    ligand_pos, receptor_pos, labels_pos = _get_positions(adata, lr_res, met=met)
    
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