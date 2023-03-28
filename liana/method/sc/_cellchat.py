import numpy as np
from liana.method._Method import Method, MethodMeta


# simplified/resource-generalizable cellchat probability score
def _lr_probability(perm_stats, axis=0):
    lr_prob = np.product(perm_stats, axis=axis)
    
    return lr_prob / (0.5 + lr_prob)  # Kh=0.5


# Internal Function to calculate CellPhoneDB LR_mean and p-values
def _cellchat_score(x, perm_stats) -> tuple:
    """
    Calculate CellChat-like LR means and p-values

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

    Returns
    -------
    A tuple with lr_mean and pvalue for x

    """
    lr_prob = _lr_probability((x['ligand_trimean'].values, x['receptor_trimean'].values))
    lr_perm_means = _lr_probability(perm_stats)
    
    # calculate p-values
    n_perms = perm_stats.shape[1]
    cellchat_pvals = np.sum(np.greater_equal(lr_perm_means, lr_prob), axis=0) / n_perms
    
    return lr_prob, cellchat_pvals


# Initialize CellChat Meta
_cellchat = MethodMeta(method_name="CellChat",
                       complex_cols=["ligand_trimean", "receptor_trimean"],
                       add_cols=['mat_max'],
                       fun=_cellchat_score,
                       magnitude="lr_probs",
                       magnitude_ascending=False,
                       specificity="cellchat_pvals",
                       specificity_ascending=True,
                       permute=True,
                       reference="Jin, S., Guerrero-Juarez, C.F., Zhang, L., Chang, I., Ramos, "
                                 "R., Kuan, C.H., Myung, P., Plikus, M.V. and Nie, Q., "
                                 "2021. Inference and analysis of cell-cell communication using "
                                 "CellChat. Nature communications, 12(1), pp.1-20. "
                       )

# Initialize callable Method instance
cellchat = Method(_SCORE=_cellchat)
