from pandas import DataFrame
from numpy import array, mean, zeros
from scipy.sparse import csr_matrix

def mean_per_cell(adata, genes):
        if genes == []:
            return zeros(adata.shape[0])
        return array(mean(adata[:,genes].X, axis=1)).flatten()


def _metalinks_estimation(me_res, adata, verbose, est_fun = mean_per_cell,) -> DataFrame: 
    """
    Estimate metabolite abundances 
    Parameters
    ----------
    me_res : pandas.core.frame.DataFrame
        metabolite-gene associations
    
    adata : anndata.AnnData
        object with gene expression data

    est_fun : function
        function to aggregate gene expression values to metabolite abundance estimates

    verbose : bool
        verbosity

    Returns
    -------
    met_est : pandas.core.frame.DataFrame
        metabolite abundance estimates

    """
    
    metabolites = me_res['HMDB'].unique()

    prod_genes = array([get_gene_sets(i, 'producing', me_res, adata.var_names) for i in metabolites], dtype=object)
    deg_genes = array([get_gene_sets(i, 'degrading', me_res, adata.var_names) for i in metabolites], dtype=object)

    prod_vals = array([est_fun(adata, prod) for prod in prod_genes])
    deg_vals = array([est_fun(adata, deg) for deg in deg_genes])

    final_estimates = get_est(prod_vals, deg_vals) # think about clippping .clip(0, None)
    
    if verbose:
        print(f"Metabolites with gene expression: {160}")
        print(f"Metabolites without gene expression: {len(metabolites) - 160}")


    return DataFrame(final_estimates, columns=adata.obs_names, index=metabolites)



def get_gene_sets(i, direction, me_res, vars):
        return get_genesets(i, me_res, vars, direction)

def get_genesets(x, df, vars, direction):
    genes = df.loc[(df['HMDB'] == x) & (df['direction'] == direction),'GENE']
    genes = [x for x in genes if x in vars]
    return genes

def get_est(prod, deg):
    return prod - deg


