#from .._ml_pipe import _get_lr_pvals 
from numpy import mean
from liana.method._pipe_utils._get_mean_perms import _calculate_pvals
import numpy as np

def _simple_prod(x, y): return (x * y) 

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
    # #exclude cell groups with no expression of ligand or receptor 
    # if (x.ligand_means == 0) | (x.receptor_means == 0):
    #     return 0, 1

    # # calculate the permutation scores
    # scores = _get_lr_pvals(x, perms_receptors, perms_ligands, ligand_pos, receptor_pos, labels_pos, _simple_prod)

    # # calculate the metalinks score
    # metalinks_score =  scores[0]/mean(perms_ligands)

    # return metalinks_score, scores[1]

    zero_msk = ((x['ligand_means'] == 0) | (x['receptor_means'] == 0))
    lr_means = np.mean((x['ligand_means'].values, x['receptor_means'].values), axis=0)
    lr_means[zero_msk] = 0
    
    cpdb_pvals = _calculate_pvals(lr_means, perm_stats, np.mean)

    return lr_means, cpdb_pvals

