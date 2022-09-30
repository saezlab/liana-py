from statsmodels.distributions.empirical_distribution import ECDF

from liana.steady.Method import Method, MethodMeta


def _simple_mean(x, y): return (x + y) / 2


# Internal Function to calculate CellPhoneDB LR_mean and p-values
def _cpdb_score(x, perms, ligand_pos, receptor_pos, labels_pos):
    if (x.ligand_means == 0) | (x.receptor_means == 0):
        return 1

    # Permutations lr mean
    ligand_perms = perms[:, labels_pos[x.source], ligand_pos[x.ligand]]
    receptor_perms = perms[:, labels_pos[x.target], receptor_pos[x.receptor]]
    lr_perms = _simple_mean(ligand_perms, receptor_perms)

    # actual lr_mean
    lr_mean = _simple_mean(x.ligand_means, x.receptor_means)

    return lr_mean, (1 - ECDF(lr_perms)(lr_mean))


_cellphonedb = MethodMeta(method_name="cellphonedb",
                          complex_cols=['ligand_means', 'receptor_means'],
                          add_cols=['ligand', 'receptor'],
                          fun=_cpdb_score,
                          magnitude='lr_means',
                          specificity='pvals',
                          permute=True,
                          reference='Efremova et al., 2020')

cellphonedb = Method(_SCORE=_cellphonedb)
