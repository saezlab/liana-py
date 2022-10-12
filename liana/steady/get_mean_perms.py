import numpy as np


def get_means_perms(adata, lr_res, n_perms, seed):
    # initialize rng
    rng = np.random.default_rng(seed=seed)

    # define labels and dict
    labels = adata.obs.label.cat.categories
    labels_dict = {label: adata.obs.label.isin([label]) for label in labels}

    # indexes to be shuffled
    idx = np.arange(adata.X.shape[0])

    # Perm should be a cube /w dims: nperms x idents x ngenes
    perms = np.zeros((n_perms, labels.shape[0], adata.shape[1]))

    # Assign permuted matrix
    for perm in range(n_perms):
        perm_idx = rng.permutation(idx)
        perm_mat = adata.X[perm_idx].copy()
        # populate matrix /w permuted means
        for cind in range(labels.shape[0]):
            perms[perm, cind] = perm_mat[labels_dict[labels[cind]]].mean(0)

    # Get indeces for each gene and label in the permutations
    ligand_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity
                  in lr_res['ligand']}
    receptor_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity
                    in lr_res['receptor']}
    labels_pos = {labels[pos]: pos for pos in range(labels.shape[0])}

    return perms, ligand_pos, receptor_pos, labels_pos
