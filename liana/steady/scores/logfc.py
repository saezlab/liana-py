from liana.steady.Method import Method, MethodMeta
from .cellphonedb import _simple_mean


def _logfc_score(x):
    # specificity
    scaled_weight = _simple_mean(x.ligand_logfc, x.receptor_logfc)
    return None, scaled_weight


# Initialize CPDB Meta
_logfc = MethodMeta(method_name="log2FC",
                    complex_cols=['ligand_means', 'receptor_means',
                                  'ligand_logfc', 'receptor_logfc'],
                    add_cols=[],
                    fun=_logfc_score,
                    magnitude=None,
                    specificity='lr_logfc',
                    permute=False,
                    reference='')

# Initialize callable Method instance
logfc = Method(_SCORE=_logfc)
