from liana.method._Method import Method, MethodMeta


def _natmi_score(x) -> tuple:
    """
    Calculate NATMI-like expression product and specificity weights

    Parameters
    ----------
    x
        DataFrame row

    Returns
    -------
    tuple(expr_prod, spec_weight)

    """

    # magnitude
    expr_prod = x.ligand_means * x.receptor_means

    # specificity
    s_lig = (x.ligand_means / x.ligand_means_sums)
    s_rec = (x.receptor_means / x.receptor_means_sums)
    spec_weight = s_lig * s_rec

    return expr_prod, spec_weight


# Initialize CPDB Meta
_natmi = MethodMeta(method_name="NATMI",
                    complex_cols=['ligand_means', 'receptor_means'],
                    add_cols=['ligand_means_sums', 'receptor_means_sums'],
                    fun=_natmi_score,
                    magnitude='expr_prod',
                    magnitude_ascending=False,
                    specificity='spec_weight',
                    specificity_ascending=False,
                    permute=False,
                    reference='Hou, R., Denisenko, E., Ong, H.T., Ramilowski, J.A. and Forrest, '
                              'A.R., 2020. Predicting cell-to-cell communication networks using '
                              'NATMI. Nature communications, 11(1), pp.1-11. '
                    )

# Initialize callable Method instance
natmi = Method(_method=_natmi)

