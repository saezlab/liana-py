from liana.steady.Method import Method, MethodMeta
from .cellphonedb import _simple_mean


def _connectome_score(x):
    # magnitude
    expr_prod = x.ligand_means * x.receptor_means
    # specificity
    scaled_weight = _simple_mean(x.ligand_zscores, x.receptor_zscores)
    return expr_prod, scaled_weight


# Initialize CPDB Meta
_connectome = MethodMeta(method_name="Connectome",
                         complex_cols=['ligand_means', 'receptor_means'],
                         add_cols=['ligand_zscores', 'receptor_zscores'],
                         fun=_connectome_score,
                         magnitude='expr_prod',
                         specificity='scaled_weight',
                         permute=False,
                         reference='')

# Initialize callable Method instance
connectome = Method(_SCORE=_connectome)
