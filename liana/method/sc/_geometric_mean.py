from scipy.stats import gmean

from liana.method._Method import Method, MethodMeta
from .._pipe_utils._get_mean_perms import _get_lr_pvals


def _gmean(x, y):
    return gmean([x, y], axis=0)


# Internal Function to calculate Geometric LR_mean and p-values
def _gmean_score(x, perms, ligand_pos, receptor_pos, labels_pos) -> tuple:
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
    return _get_lr_pvals(x, perms, ligand_pos, receptor_pos, labels_pos, _gmean)


# Initialize Meta
_geometric_mean = MethodMeta(method_name="Geometric Mean",
                             complex_cols=["ligand_means", "receptor_means"],
                             add_cols=[],
                             fun=_gmean_score,
                             magnitude="lr_gmeans",
                             magnitude_ascending=False,
                             specificity="pvals",
                             specificity_ascending=True,
                             permute=True,
                             reference="CellPhoneDBv2's permutation approach applied to the "
                                       "geometric means of ligand-receptors, as opposed to "
                                       "their arithmetic mean."
                             )

# Initialize callable Method instance
geometric_mean = Method(_method=_geometric_mean)
