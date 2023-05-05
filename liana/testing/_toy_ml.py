from numpy import random
from pandas import DataFrame
from scanpy.datasets import pbmc68k_reduced
from ._sample_lrs import sample_mrs

def get_toy_ml():
    adata = pbmc68k_reduced()
    sample_key = 'sample'
    
    rng = random.default_rng(0)

    # add fake metabolite data
    adata.obsm['metabolite_abundance'] = rng.random((adata.n_obs, 5))
    adata.uns['CCC_res'] = sample_mrs(by_sample=True)
    # add a mask with 0 and 1 with in the shape of genes x mtabolites
    adata.uns['mask'] = DataFrame(rng.choice([-1, 0, 1], size=(adata.n_vars, 5)), index=adata.var_names, columns = ['A', 'B', 'C', 'D', 'E'])
    
    return adata
