from liana.steady.Method import Method, MethodMeta
from .cellphonedb import _simple_mean


def _logfc_score(x):
    # specificity
    scaled_weight = _simple_mean(x.ligand_logfc, x.receptor_logfc)
    return None, scaled_weight


# Initialize CPDB Meta
_logfc = MethodMeta(method_name="Connectome",
                    complex_cols=['ligand_means', 'receptor_means'],
                    add_cols=['ligand_logfc', 'receptor_logfc'],
                    fun=_logfc_score,
                    magnitude=None,
                    specificity='lr_logfc',
                    permute=False,
                    reference='')

# Initialize callable Method instance
logfc = Method(_SCORE=_logfc)
