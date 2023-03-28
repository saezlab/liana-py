from __future__ import annotations

import anndata
import numpy as np
import pandas
from tqdm import tqdm

def _get_means_perms(adata: anndata.AnnData,
                     n_perms: int,
                     seed: int,
                     agg_fun,
                     norm_factor: float | None,
                     verbose: bool):
    """
    Generate permutations and indices required for permutation-based methods

    Parameters
    ----------
    adata
        Annotated data matrix.
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
            ct_mask = labels_dict[labels[cind]]
            perms[perm, cind] = agg_fun(perm_mat[ct_mask], axis=0)

    return perms

def _get_positions(adata, lr_res):
    labels = adata.obs['label'].cat.categories
    
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
