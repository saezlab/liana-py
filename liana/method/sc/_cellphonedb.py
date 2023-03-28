import numpy as np

from liana.method._Method import Method, MethodMeta
from liana.method._pipe_utils._get_mean_perms import _get_lr_pvals


def _simple_mean(x, y): return (x + y) / 2


# Internal Function to calculate CellPhoneDB LR_mean and p-values
def _cpdb_score(x, perms, ligand_pos, receptor_pos, labels_pos) -> tuple:
    """
    Calculate CellPhoneDB-like LR means and p-values
    
    Parameters
    ----------
    x
        DataFrame with LIANA results
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
    A tuple with lr_mean and p-value for x

    """
    zero_msk = ((x['ligand_means'] == 0) | (x['receptor_means'] == 0))
    lr_means = np.mean((x['ligand_means'].values, x['receptor_means'].values), axis=0)
    lr_means[zero_msk] = 0
    
    # we have lr_scores
    
    # we want to now get permutated lr_scores
    # all at the same time
    ligand_idx = x['ligand'].map(ligand_pos)
    receptor_idx = x['receptor'].map(receptor_pos)
    source_idx = x['source'].map(labels_pos)
    target_idx = x['target'].map(labels_pos)
    
    ligand_perm_means = perms[:, source_idx, ligand_idx]
    receptor_perm_means = perms[:, target_idx, receptor_idx]
    lr_perm_means = (ligand_perm_means + receptor_perm_means) / 2
    
    # calculate p-values
    n_perms = perms.shape[0]
    p_values = np.sum(np.greater_equal(lr_perm_means, lr_means), axis=0) / n_perms

    return lr_means, p_values


# Initialize CPDB Meta
_cellphonedb = MethodMeta(method_name="CellPhoneDB",
                          complex_cols=["ligand_means", "receptor_means"],
                          add_cols=[],
                          fun=_cpdb_score,
                          magnitude="lr_means",
                          magnitude_ascending=False,
                          specificity="cellphone_pvals",
                          specificity_ascending=True,
                          permute=True,
                          reference="Efremova, M., Vento-Tormo, M., Teichmann, S.A. and "
                                    "Vento-Tormo, R., 2020. CellPhoneDB: inferring cell–cell "
                                    "communication from combined expression of multi-subunit "
                                    "ligand–receptor complexes. Nature protocols, 15(4), "
                                    "pp.1484-1506. "
                          )

# Initialize callable Method instance
cellphonedb = Method(_SCORE=_cellphonedb)
