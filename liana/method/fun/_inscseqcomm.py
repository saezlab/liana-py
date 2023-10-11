from liana._logging import _logg
import liana as li
import pandas as pd
import scanpy as sc
from liana.method._pipe_utils import prep_check_adata








if __name__ == "__main__":
    
    adata = li.testing.datasets.kang_2018()
    pd.set_option('display.max_colwidth', 20)
    print("DEBUG: ad | describe: {}".format(adata))
    print("DEBUG: ad | shape: {}".format(adata.shape))
    print("DEBUG: ad | obs_keys: {}".format(adata.obs.keys()))
    print("DEBUG: ad | var_keys: {}".format(adata.var.keys()))
    print("DEBUG: ad | obs head: \n{}".format(adata.obs.head().astype(str)))
    print("DEBUG: ad | var head: \n{}".format(adata.var.head().astype(str)))
    
    
    
    
    sample_key = 'sample'
    groupby = 'cell_type'
    condition_key = 'condition'
    min_cells = 5
    use_raw = False
    layer = None
    verbose = True
    
    print("DEBUG: ad | normalized counts shape: {} and type {}".format(adata.X.shape, type(adata.X)))
    print("DEBUG: ad | transposed normalized counts shape: {} and type {}".format(adata.X.T.shape, type(adata.X.T)))
    
    adata = prep_check_adata(adata,
                             groupby=groupby,
                             min_cells=min_cells,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose)