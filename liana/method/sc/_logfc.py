from liana.method._Method import Method, MethodMeta
from ._cellphonedb import _simple_mean


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
                    magnitude_ascending=None,
                    specificity='lr_logfc',
                    specificity_ascending=False,
                    permute=False,
                    reference='Dimitrov, D., Türei, D., Garrido-Rodriguez, M., Burmedi, P.L., '
                              'Nagai, J.S., Boys, C., Ramirez Flores, R.O., Kim, H., Szalai, B., '
                              'Costa, I.G. and Valdeolivas, A., 2022. Comparison of methods and '
                              'resources for cell-cell communication inference from single-cell '
                              'RNA-Seq data. Nature Communications, 13(1), pp.1-13. '
                    )

# Initialize callable Method instance
logfc = Method(_method=_logfc)
