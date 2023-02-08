from liana.method.ml._ml_Method import MetabMethod, MetabMethodMeta
from ..._pipe_utils._get_mean_perms import _get_lr_pvals 
from numpy import mean

# define a function that calculates the product of two numbers
def _simple_prod(x, y): return (x * y) 

# write function that calculates the mebocost score from transcriptome and metabolome data
def _mebocost_score(x, perms, ligand_pos, receptor_pos, labels_pos) -> tuple: 
    """
    infer mebocost-like CCC from metabolite and transcript abundance:
    
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
    #exclude cell groups with no expression of ligand or receptor
    if (x.ligand_means == 0) | (x.receptor_means == 0):
        return 0, 1

    # calculate the permutation scores
    scores = _get_lr_pvals(x, perms, ligand_pos, receptor_pos, labels_pos, _simple_prod)

    # calculate the mebocost score
    mebocost_score =  scores[0]/mean(perms)

    return mebocost_score, scores[1]

# add fdr and expr_prop to the mebocost score !!!!


# Initialize mebocost Meta
_mebocost = MetabMethodMeta(est_method_name="MEBOCOST_EST",
                            score_method_name="MEBOCOST",
                            complex_cols=['ligand_means', 'receptor_means'],
                            add_cols=['ligand_means_sums', 'receptor_means_sums'],
                            fun=_mebocost_score,
                            magnitude='mebocost_score',
                            magnitude_ascending=False, 
                            specificity='cellphone_pval',
                            specificity_ascending=True,  
                            permute=True,
                            agg_fun=_simple_prod, #check again
                            score_reference='Zheng, R., Zhang, Y., Tsuji, T., Zhang, L., Tseng, Y.-H.& Chen,'
                                    'K., 2022,“MEBOCOST: Metabolic Cell-Cell Communication Modeling '
                                    'by Single Cell Transcriptome,” BioRxiv.',
                            est_reference=None
                            
                    )

# Initialize callable Method instance
mebocost = MetabMethod(_SCORE=_mebocost)




