from liana.method.sc._Method import Method, MethodMeta
from liana.method._pipe_utils._get_mean_perms import _calculate_pvals
import liana as li
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import norm
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=np.inf)

"""
def _gene_score(gene_mean, cluster_mean, cluster_std, cluster_counts):
    probability = norm.cdf(gene_mean, loc=cluster_mean, scale = cluster_std/np.sqrt(cluster_counts))
    return probability

def _lr_score(perm_stats, axis=0):
    inter_score = np.minimum(perm_stats[0],perm_stats[1])
    return inter_score

def _intercellular_score(x, perm_stats):
    ligand_score = _gene_score(x['ligand_means'], x['source_cluster_mean'], x['source_cluster_std'], x['source_cluster_counts'])
    x["ligand_score"] = ligand_score
    receptor_score = _gene_score(x['receptor_means'], x['target_cluster_mean'], x['target_cluster_std'], x['target_cluster_counts'])
    x["receptor_score"] = receptor_score
    filt_df = x[x['receptor'] =='CD8B']
    print(filt_df['receptor_means'])
    inter_score = _lr_score((ligand_score, receptor_score))
    print(x.columns)
    #print(x.to_string(max_colwidth=10))
    input("SONO QUI")
    print("These are the scores: \n{}".format(inter_score))
    print("maximum inter_score: {}".format(np.max(inter_score)))   
    print("These are the shapes: Expected: \nFound: \n{}\n{}".format(inter_score.shape,
                                                                        perm_stats.shape))
    import matplotlib.pyplot as plt

    # Create a histogram
    plt.hist(inter_score, bins=10, edgecolor='black')  # Adjust the number of bins as needed
    plt.title('Histogram')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Show the histogram
    plt.show()

    return inter_score
"""

def _lr_score(perm_stats, axis=0):
    inter_score = np.minimum(perm_stats[0],perm_stats[1])
    return inter_score
    
def _intercellular_score(x, perm_stats):
    inter_score = _lr_score((x['ligand_score'], x['receptor_score']))
    
    print("SHAPE OF LR_RES {}".format(x.shape))
    filtered_df = x.copy()
    filtered_df = x[(x['receptor'].str.contains('CD8A|CD8B', regex=True))]
    filtered_df = filtered_df[['source','target','receptor','receptor_complex','receptor_means','receptor_score', 'receptor_props']]
    print("SHAPE {}\n{}".format(filtered_df.shape,filtered_df.to_string(max_colwidth=20)))
    input("Inside intercellular score function...")
    #print(x.to_string(max_colwidth=10))
    print("These are the scores: \n{}".format(inter_score))
    print("maximum inter_score: {}".format(np.max(inter_score)))   
    print("These are the shapes: Expected: \nFound: \n{}\n{}".format(inter_score.shape,
                                                                        perm_stats.shape))
    import matplotlib.pyplot as plt

    # Create a histogram
    plt.hist(inter_score, bins=10, edgecolor='black')  # Adjust the number of bins as needed
    plt.title('Histogram')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Show the histogram
    plt.show()

    return inter_score


if __name__ == "__main__":
    ad = sc.datasets.pbmc68k_reduced()
    # Initialize scSeqComm Meta
    foo = MethodMeta(method_name="scSeqComm",
                        complex_cols=["ligand_means","receptor_means"],
                        add_cols=["ligand_score","receptor_score"],
                        fun=_intercellular_score,
                        magnitude="inter_score",
                        magnitude_ascending=False,
                        specificity=None,
                        specificity_ascending=True,
                        permute=True,
                        reference="Baruzzo, G., Cesaro, G., Di Camillo, B. "
                                  "2022. Identify, quantify and characterize cellular communication "
                                  "from single-cell RNA-sequencing data with scSeqComm. Bioinformatics, "
                                  "38(7), pp.1920-1929"
                        )

    #_scseqcomm = Method(_method=_scseqcomm)
    foo = Method(_method=foo)
    foo(adata=ad,groupby='bulk_labels',use_raw=True, expr_prop=0.1, return_all_lrs=True, key_added='scseqcomm_res')
    

    """     
    p = li.pl.dotplot(adata=ad,
                  colour='inter_score',
                  size='inter_score',
                  source_labels=['CD34+', 'CD56+ NK', 'CD14+ Monocyte', 'Dendritic'],
                  target_labels=['CD34+','CD56+ NK', 'Dendritic'],
                  figure_size=(8,7),
                  uns_key='scseqcomm_res')
    print(p) 
    """
    
    
