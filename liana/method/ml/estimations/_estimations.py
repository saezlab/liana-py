from pandas import DataFrame
from numpy import array, mean, zeros, median, diff, divide, zeros_like, in1d
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import gmean, hmean
from tqdm import tqdm
from decoupler import run_ulm, run_mlm


def mean_per_cell(adata, genes):
        if genes == []:
            return zeros(adata.shape[0])
        return array(mean(adata[:,genes].X, axis=1)).flatten()

def nnzmean_per_cell(adata, genes):
        if genes == []:
            return zeros(adata.shape[0])
        a = adata[:,genes].X
        sums = a.sum(axis=1).A1
        counts = diff(a.indptr)
        averages = divide(sums, counts, out=zeros_like(sums), where=counts!=0)
        return averages

def max_per_cell(adata, genes):
        if genes == []:
            return zeros(adata.shape[0])
        a = adata[:,genes].X.max(axis=1).toarray().flatten()
        return a

def gmean_per_cell(adata, genes):
        if genes == []:
            return zeros(adata.shape[0])
        return array(gmean(adata[:,genes].X)).flatten()

def hmean_per_cell(adata, genes):
        if genes == []:
            return zeros(adata.shape[0])
        return array(hmean(adata[:,genes].X)).flatten()


def _metalinks_estimation(me_res, adata, verbose, est_fun = 'mean_per_cell') -> DataFrame: 
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

    # write dictionary that links est_fun to method name
    # est_fun_dict = {
    #                 # 'mean_per_cell': mean_per_cell,
    #                 # 'nnzmean_per_cell': nnzmean_per_cell,
    #                 # 'gmean_per_cell': gmean_per_cell,
    #                 # 'hmean_per_cell': hmean_per_cell,
    #                 # 'max_per_cell': max_per_cell,
    #                 'ulm': 'ulm'
    #                 }

    # if est_fun not in est_fun_dict.keys():
    #     raise ValueError(f"est_fun must be one of {est_fun_dict.keys()}")

    # est_fun = est_fun_dict[est_fun]
    
    metabolites = me_res['HMDB'].unique()

    metabolites.sort()

    mask = lil_matrix((len(metabolites), adata.shape[1]))

    mask.index = metabolites
    mask.columns = adata.var_names

    met_bool = array([False] * len(metabolites))
    gene_bool = array([False] * adata.shape[1])

    for _, row in tqdm(me_res.iterrows()):
        met_bool = metabolites == row['HMDB']
        gene_bool = adata.var_names == row['GENE']
        mask[met_bool, gene_bool] = 1 if row['direction'] == 'producing' else -1

    mask = mask.tocsr()

    if est_fun in ['ulm', 'mlm']:

        me_res['weight'] = me_res['direction'].apply(lambda x: 1 if x == 'producing' else -1)
        me_res.drop_duplicates(subset=['HMDB', 'GENE'], inplace=True) ## attention: decide on direction 

        if est_fun == 'ulm':
            run_ulm(adata, me_res, source = 'HMDB',  target = 'GENE', weight = 'weight', verbose=verbose, use_raw=False, min_n = 1)
            estimates = adata.obsm['ulm_estimate']
            estimates[estimates < 0] = 0
            final_estimates = csr_matrix(estimates)
            metabolites_estimated = adata.obsm['ulm_estimate'].columns
            
        elif est_fun == 'mlm': # get it working !
            run_mlm(adata, me_res, source = 'HMDB',  target = 'GENE', weight = 'weight', verbose=verbose, use_raw=False, min_n = 1)
            estimates = adata.obsm['mlm_estimate']
            estimates[estimates < 0] = 0
            final_estimates = csr_matrix(estimates)
            metabolites_estimated = adata.obsm['mlm_estimate'].columns

        mask = mask[in1d(metabolites, metabolites_estimated), :]

    else:

        estimates = adata.X.dot(mask.T)
        estimates[estimates < 0] = 0
        metabolites_estimated = metabolites[estimates.getnnz(0) > 0]
        mask = mask[ estimates.getnnz(0) > 0,:]
        final_estimates = estimates[:, estimates.getnnz(0) > 0]

    if verbose:
        print(f"Metabolites with final estimates: {final_estimates.shape[1]} \n")

    return final_estimates, metabolites_estimated, mask



# def get_gene_sets(i, direction, me_res, vars):
#         return get_genesets(i, me_res, vars, direction)

# def get_genesets(x, df, vars, direction):
#     genes = df.loc[(df['HMDB'] == x) & (df['direction'] == direction),'GENE']
#     genes = [x for x in genes if x in vars]
#     return genes

# def get_est(prod, deg):
#     return prod - deg


