from liana.method.sc._Method import Method, MethodMeta
from numpy import mean

def _logfc_score(x):
    mean_logfc = mean((x['ligand_logfc'], x['receptor_logfc']), axis=0)
    return None, mean_logfc


# Initialize CPDB Meta
_logfc = MethodMeta(method_name="log2FC",
                    complex_cols=['ligand_means', 'receptor_means'],
                    add_cols=['ligand_logfc', 'receptor_logfc'],
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

logfc = Method(_method=_logfc)
