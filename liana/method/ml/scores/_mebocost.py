from liana.method._Method import Method, MethodMeta
from ..._pipe_utils._get_mean_perms import _get_lr_pvals # from cellphonedb
from numpy import mean
# write function that efficiently reads in a dataframe from csv and returns a dataframe using numpy

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

    # calculate the permu
    scores = _get_lr_pvals(x, perms, ligand_pos, receptor_pos, labels_pos, _simple_prod)

    # calculate the mebocost score
    mebocost_score =  scores[0]/mean(perms)

    return mebocost_score, scores[1]




# Initialize mebocost Meta
_mebocost = MethodMeta(method_name="MEBOCOST",
                    complex_cols=['ligand_means', 'receptor_means'], ## attention
                    add_cols=['ligand_means_sums', 'receptor_means_sums'], ## attention
                    fun=_mebocost_score,
                    magnitude='mebocost_score', ## attention
                    magnitude_ascending=False,  ## attention
                    specificity='cellphone_pval', ## attention
                    specificity_ascending=True,  ## attention
                    permute=True,
                    reference='Zheng, R., Zhang, Y., Tsuji, T., Zhang, L., Tseng, Y.-H.& Chen,'
                              'K., 2022,“MEBOCOST: Metabolic Cell-Cell Communication Modeling '
                              'by Single Cell Transcriptome,” BioRxiv.'
                    )

# Initialize callable Method instance
mebocost = Method(_SCORE=_mebocost)




