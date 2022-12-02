from liana.method._Method import Method, MethodMeta
from ._cellphonedb import _simple_mean


def _connectome_score(x) -> tuple:
    """
    Calculate Connectome-like Score

    Parameters
    ----------
    x
        DataFrame row

    Returns
    -------
    tuple(expr_prod, scaled_weight)

    """
    # magnitude
    expr_prod = x.ligand_means * x.receptor_means
    # specificity
    scaled_weight = _simple_mean(x.ligand_zscores, x.receptor_zscores)
    return expr_prod, scaled_weight


# Initialize CPDB Meta
_connectome = MethodMeta(method_name="Connectome",
                         complex_cols=['ligand_zscores', 'receptor_zscores',
                                       'ligand_means', 'receptor_means'],
                         add_cols=[],
                         fun=_connectome_score,
                         magnitude='expr_prod',
                         magnitude_ascending=False,
                         specificity='scaled_weight',
                         specificity_ascending=False,
                         permute=False,
                         reference='Raredon, M.S.B., Yang, J., Garritano, J., Wang, M., Kushnir, '
                                   'D., Schupp, J.C., Adams, T.S., Greaney, A.M., Leiby, K.L., '
                                   'Kaminski, N. and Kluger, Y., 2022. Computation and '
                                   'visualization of cell–cell signaling topologies in '
                                   'single-cell systems data using Connectome. Scientific '
                                   'reports, 12(1), pp.1-12. '
                         )

# Initialize callable Method instance
connectome = Method(_method=_connectome)
