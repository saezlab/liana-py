from liana.method._Method import Method, MethodMeta
from .._pipe_utils._get_mean_perms import _get_lr_pvals


def _simple_mean(x, y): return (x + y) / 2


# Internal Function to calculate CellPhoneDB LR_mean and p-values
def _cpdb_score(x, perms, ligand_pos, receptor_pos, labels_pos) -> tuple:
    """
    Calculate CellPhoneDB-like LR means and p-values
    
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
    if (x.ligand_means == 0) | (x.receptor_means == 0):
        return 0, 1

    return _get_lr_pvals(x, perms, ligand_pos, receptor_pos, labels_pos, _simple_mean)


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
cellphonedb = Method(_method=_cellphonedb)
