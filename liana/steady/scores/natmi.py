from liana.steady.Method import Method, MethodMeta


def _natmi_score(x):
    # magnitude
    expr_prop = x.ligand_means * x.receptor_means

    # specificity
    s_lig = (x.ligand_means / x.ligand_means_sums)
    s_rec = (x.receptor_means / x.receptor_means_sums)
    spec_weight = s_lig * s_rec

    return expr_prop, spec_weight


# Initialize CPDB Meta
_natmi = MethodMeta(method_name="natmi",
                    complex_cols=['ligand_means', 'receptor_means'],
                    add_cols=['ligand', 'receptor',
                              'ligand_means_sums', 'receptor_means_sums'],
                    fun=_natmi_score,
                    magnitude='expr_prod',
                    specificity='spec_weight',
                    permute=False,
                    reference='Hou et al., 2021')

# Initialize callable Method instance
natmi = Method(_SCORE=_natmi)

