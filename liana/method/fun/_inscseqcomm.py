from liana._logging import _logg
import liana as li
import pandas as pd
import scanpy as sc
import numpy as np
import decoupler as dc
from liana.method._pipe_utils import prep_check_adata
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


#def _wilcoxon_de():

if __name__ == "__main__":
    
    adata = li.testing.datasets.kang_2018()
    pd.set_option('display.max_colwidth', 20)
    
    sample_key = 'sample'
    groupby = 'cell_type'
    condition_key = 'condition'
    min_cells = 5
    use_raw = False
    layer = None
    verbose = True
    print(adata)
    print(adata.var['name'].unique())
    adata.X = adata.X.astype(np.int64)
    print(np.max(adata.X))

    print(all(isinstance(val, np.int64) for val in adata.X.data)) # False
    
    # Iterate through the data and find non-integer elements
    non_integer_elements = []
    for row, col, val in zip(adata.X.nonzero()[0], adata.X.nonzero()[1], adata.X.data):
        if not isinstance(val, np.int64):
            print(type(val))
            non_integer_elements.append((row, col, val))
    print(non_integer_elements)
    
        
    print("DEBUG: ad | normalized counts shape: {} and type {}".format(adata.X.shape, type(adata.X)))
    print("DEBUG: ad | transposed normalized counts shape: {} and type {}".format(adata.X.T.shape, type(adata.X.T)))
    
    tf_tg_list = li.rs.select_resource_tf_tg()
    rec_tf_ppr_list = li.rs.select_resource_ppr_r_tf()
    
    pdata = dc.get_pseudobulk(adata,
                              sample_col = sample_key,
                              groups_col = groupby,
                              layer = 'counts',
                              mode = 'mean',
                              min_cells=10,
                              min_counts=1e4)
    """
    p = dc.plot_psbulk_samples(pdata, groupby=[sample_key, groupby], figsize=(11, 4), return_fig=True)
    import matplotlib.pyplot as plt
    plt.show()
    """
    # Differential expression anaalysis scection
    dea_results = {}
    for cell_group in pdata.obs[groupby].unique():
        # Select cell profiles
        ctdata = pdata[pdata.obs[groupby] == cell_group].copy()

        # Obtain genes that pass the edgeR-like thresholds
        # NOTE: QC thresholds might differ between cell types, consider applying them by cell type
        genes = dc.filter_by_expr(ctdata,
                                group=condition_key,
                                min_count=5, # a minimum number of counts in a number of samples
                                min_total_count=10 # a minimum total number of reads across samples
                                )

        # Filter by these genes
        ctdata = ctdata[:, genes].copy()
        
        # Build DESeq2 object
        # NOTE: this data is actually paired, so one could consider fitting the patient label as a confounder
        dds = DeseqDataSet(
            adata=ctdata,
            design_factors=condition_key,
            ref_level=[condition_key, 'ctrl'], # set control as reference
            refit_cooks=True,
            n_cpus=None,
        )
        
        # Compute LFCs
        dds.deseq2()
        # Contrast between stim and ctrl
        stat_res = DeseqStats(dds, contrast=[condition_key, 'stim', 'ctrl'], n_cpus=8)
        # Compute Wald test
        stat_res.summary()
        # Shrink LFCs
        stat_res.lfc_shrink(coeff='condition_stim_vs_ctrl') # {condition_key}_cond_vs_ref
        
        dea_results[cell_group] = stat_res.results_df
    
    print(dea_results)