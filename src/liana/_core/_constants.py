from numpy import exp, finfo, log10


class DefaultValues:
    """Default Values"""

    logbase = exp(1)
    min_cells = 5
    expr_prop = 0.1
    n_perms = 1000
    seed = 1337
    de_method = 't-test'
    resource_name = 'consensus'
    resource = None
    interactions = None
    layer = None
    use_raw = True
    verbose = False
    return_all_lrs = False
    supp_columns = None
    inplace = True
    groupby_pairs = None

    return_fig = True
    cmap = 'viridis'

    lr_sep = '^'
    complex_sep = "_"

    def inverse_fun(x):
        return -log10(x + finfo(float).eps)

class Keys:
    """Keys related to AnnData"""

    uns_key = 'liana_res'
    spatial_key = 'spatial'
    connectivity_key = f'{spatial_key}_connectivities'
    target_metrics = 'target_metrics'
    interactions = 'interactions'

class PrimaryColumns:
    source = 'source'
    target = 'target'
    ligand = 'ligand'
    receptor = 'receptor'
    ligand_complex = 'ligand_complex'
    receptor_complex = 'receptor_complex'
    primary = [source, target, ligand_complex, receptor_complex]
    complete = primary + [ligand, receptor]

class CommonColumns:
    ligand_means = 'ligand_means'
    receptor_means = 'receptor_means'
    ligand_props = 'ligand_props'
    receptor_props = 'receptor_props'
    ligand_pvals = 'ligand_pvals'
    receptor_pvals = 'receptor_pvals'

class MethodColumns:
    ligand_means_sums = 'ligand_means_sums'
    receptor_means_sums = 'receptor_means_sums'
    ligand_zscores = 'ligand_zscores'
    receptor_zscores = 'receptor_zscores'
    ligand_logfc = 'ligand_logfc'
    receptor_logfc = 'receptor_logfc'
    ligand_trimean = 'ligand_trimean'
    receptor_trimean = 'receptor_trimean'
    mat_mean = 'mat_mean'
    mat_max = 'mat_max'
    ligand_cdf = 'ligand_cdf'
    receptor_cdf = 'receptor_cdf'

    @classmethod
    def get_all_values(cls):
        return [value for name, value in cls.__dict__.items()
                if not name.startswith('__') and isinstance(value, str)]

class InternalValues:
    lrs_to_keep = 'lrs_to_keep'
    prop_min = 'prop_min'
    label = '@label'
