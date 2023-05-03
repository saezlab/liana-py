from pandas import DataFrame
from numpy import array, in1d
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
from decoupler import run_ulm, run_mlm, run_wmean, run_wsum, run_mdt, run_udt, run_viper, run_ora, run_gsea, run_gsva


def _metalinks_estimation(me_res, adata, verbose, est_fun = 'mean_per_cell', pass_mask = False, **kwargs) -> DataFrame: 
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

    if verbose:
        print(f"Estimating metabolite abundances with {est_fun}...")

    est_fun_dict = {'ulm': run_ulm,
                    'mlm': run_mlm,
                    'wmean': run_wmean,
                    'wsum': run_wsum,
                    'udt': run_udt,
                    'mdt': run_mdt,
                    'viper': run_viper,
                    'ora': run_ora,
                    'gsea': run_gsea,
                    'gsva': run_gsva
    }

    
    metabolites = me_res['HMDB'].unique()

    metabolites.sort()

    

    mask = lil_matrix((len(metabolites), adata.shape[1]))

    mask.index = metabolites
    mask.columns = adata.var_names

    if est_fun == 'mean_per_cell':
        pass_mask = True

    if pass_mask:

        if verbose:
            print('Preparing mask...')

        met_bool = array([False] * len(metabolites))
        gene_bool = array([False] * adata.shape[1])

        for _, row in tqdm(me_res.iterrows()):
            met_bool = metabolites == row['HMDB']
            gene_bool = adata.var_names == row['GENE']
            mask[met_bool, gene_bool] = 1 if row['direction'] == 'producing' else -1

    mask = mask.tocsr()

    if est_fun in est_fun_dict.keys():

        fun = est_fun_dict[est_fun]

        me_res['weight'] = me_res['direction'].apply(lambda x: 1 if x == 'producing' else -1)
        me_res.drop_duplicates(subset=['HMDB', 'GENE'], inplace=True) ## attention: decide on direction 
        df = DataFrame(adata.X.todense(), index = adata.obs_names, columns = adata.var_names)
        estimates = fun(df, me_res, source = 'HMDB',  target = 'GENE', weight = 'weight', verbose=verbose, use_raw=False, **kwargs)
        estimates = estimates[len(estimates) - 1]
        estimates[estimates < 0] = 0
        final_estimates = csr_matrix(estimates)
        metabolites_estimated = estimates.columns
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

