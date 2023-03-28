import numpy as np
from liana.method._Method import Method, MethodMeta

# Internal Function to calculate CellPhoneDB LR_mean and p-values
def _cpdb_score(x, perm_stats) -> tuple:
    """
    Calculate CellPhoneDB-like LR means and p-values
    
    Parameters
    ----------
    x
        DataFrame with LIANA results
    perms
        3D tensor with permuted averages per cluster
    ligand_idx
        Index of the ligand in the perms tensor
    receptor_idx
        Index of the receptor in the perms tensor
    source_idx
        Index of the source cell identity in the perms tensor
    target_idx
        Index of the target cell identity in the perms tensor

    Returns
    -------
    A tuple with lr_mean and p-value for x

    """
    zero_msk = ((x['ligand_means'] == 0) | (x['receptor_means'] == 0))
    lr_means = np.mean((x['ligand_means'].values, x['receptor_means'].values), axis=0)
    lr_means[zero_msk] = 0
    lr_perm_means = np.mean(perm_stats, axis=0)
    
    # calculate p-values
    n_perms = perm_stats.shape[1]
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
