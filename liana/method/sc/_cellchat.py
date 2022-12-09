from liana.method._Method import Method, MethodMeta
from .._pipe_utils._get_mean_perms import _get_lr_pvals


# simplified/resource-generalizable cellchat probability score
def _lr_probability(ligand_trimean, receptor_trimean):
    lr_prob = ligand_trimean * receptor_trimean
    return lr_prob / (0.5 + lr_prob)  # Kh=0.5


# Internal Function to calculate CellPhoneDB LR_mean and p-values
def _cellchat_score(x, perms, ligand_pos, receptor_pos, labels_pos) -> tuple:
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
    return _get_lr_pvals(x=x,
                         perms=perms,
                         ligand_pos=ligand_pos,
                         receptor_pos=receptor_pos,
                         labels_pos=labels_pos,
                         agg_fun=_lr_probability,
                         ligand_col='ligand_trimean',
                         receptor_col='receptor_trimean')


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
cellchat = Method(_method=_cellchat)
