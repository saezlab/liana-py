from liana.steady.Method import Method, MethodMeta
from .cellphonedb import _simple_mean


def _logfc_score(x):
    """
    Calculate 1vs Rest expression logFC

    Parameters
    ----------
    x
        DataFrame row

    Returns
    -------
    (None, 1vsRest expression logFC)

    """
    # specificity
    mean_logfc = _simple_mean(x.ligand_logfc, x.receptor_logfc)
    return None, mean_logfc


# Initialize CPDB Meta
_logfc = MethodMeta(method_name="log2FC",
                    complex_cols=['ligand_means', 'receptor_means',
                                  'ligand_logfc', 'receptor_logfc'],
                    add_cols=[],
                    fun=_logfc_score,
                    magnitude=None,
                    magnitude_desc=None,
                    specificity='lr_logfc',
                    specificity_desc=True,
                    permute=False,
                    reference=''
                    )

# Initialize callable Method instance
logfc = Method(_SCORE=_logfc)
