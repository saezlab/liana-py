from .._Method import MethodMeta, MetabMethod
from ...method._pipe_utils._get_mean_perms import _calculate_pvals
import numpy as np

#TODO: remove this
def _product_score(x, perm_stats) -> tuple: 
    """
    infer CCC from metabolite and transcript abundance:
    
    Parameters
    ----------
    x
        DataFrame row
    perms   
        3D tensor with permuted averages per cluster    
    ligand_pos      
        Index of the ligand in the tensor
    receptor_pos
        Index of the receptor in the perms tensor
    labels_pos              
        Index of cell identities in the perms tensor
        
    Returns
    -------
    tuple(MR_interaction, confidence_score)
    
    """

    zero_msk = ((x['ligand_means'] == 0) | (x['receptor_means'] == 0))
    lr_means = np.mean((x['ligand_means'].values, x['receptor_means'].values), axis=0)
    lr_means[zero_msk] = 0
    
    cpdb_pvals = _calculate_pvals(lr_means, perm_stats, np.mean)

    return lr_means, cpdb_pvals

# Initialize metalinks Meta
_metalinks = MethodMeta(method_name="metalinks",
                            complex_cols=['ligand_means', 'receptor_means'],
                            add_cols=['ligand_name'],
                            fun=_product_score,
                            magnitude='metalinks_score',
                            magnitude_ascending=False, 
                            specificity='pval',
                            specificity_ascending=True,  
                            permute=True,
                            reference='LIANA, Decoupler or scores',
                            met = True,
                    )

# Initialize callable Method instance
metalinks = MetabMethod(_method=_metalinks)
