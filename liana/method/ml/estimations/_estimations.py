from pandas import DataFrame
from numpy import array, in1d, vstack, array
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
from decoupler import run_ulm, run_wmean


def metalinks_estimation(me_res, adata, verbose, est_fun = 'mean_per_cell', pass_mask = False, **kwargs) -> DataFrame: 
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
                    'wmean': run_wmean
    }

    
    metabolites = me_res['HMDB'].unique()

    metabolites.sort()

    if est_fun == 'transport':

        # replace nas with unknown
        me_res['transport'].fillna('unknown', inplace=True)
        me_res['transport_direction'].fillna('unknown', inplace=True)
        me_res['reversibility'].fillna('unknown', inplace=True)

        # split me_res in two dataframes, transport und PD by the columns transport. if unknown assign to PD
        PD = me_res[me_res['transport'] == 'unknown']
        transport = me_res[me_res['transport'] != 'unknown']

        transport_out = transport[~((transport['transport_direction'] == 'in') & (transport['reversibility'] == 'reversible'))]
        transport_in = transport[~((transport['transport_direction'] == 'out') & (transport['reversibility'] == 'reversible'))]


    mask = lil_matrix((len(metabolites), adata.shape[1]))

    mask.index = metabolites
    mask.columns = adata.var_names

    if pass_mask or est_fun == 'mean_per_cell':

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
        
        final_estimates, mask, metabolites_estimated = process_estimation(est_fun, est_fun_dict,
                                                                          me_res, adata, mask, 
                                                                          metabolites, verbose, **kwargs)

    elif est_fun == 'transport':
    
        final_estimates, mask, metabolites_estimated = process_estimation(est_fun, est_fun_dict,
                                                                          me_res, adata, mask, metabolites,
                                                                          transport_out, transport_in,
                                                                          verbose, **kwargs)
    
    else:

        estimates = adata.X.dot(mask.T)
        estimates = estimates / mask.getnnz(axis=1)
        estimates[estimates != estimates] = 0
        estimates[estimates < 0] = 0
        estimates = csr_matrix(estimates)
        metabolites_estimated = metabolites[estimates.getnnz(0) > 0]
        mask = mask[estimates.getnnz(0) > 0,:]
        final_estimates = estimates[:, estimates.getnnz(0) > 0]

    if verbose:
        print(f"Metabolites with final estimates: {final_estimates.shape[1]} \n")

    return final_estimates, metabolites_estimated, mask



def process_estimation(est_fun, est_fun_dict, me_res, adata,  mask, metabolites, transport_out = None, transport_in = None, verbose=True, **kwargs):
    def apply_weight(x):
        return 1 if x == 'producing' else -1

    def estimate_values(df, res, source, target, weight, verbose, use_raw, **kwargs):
        estimates = fun(df, res, source=source, target=target, weight=weight, verbose=verbose, use_raw=use_raw, **kwargs)
        estimates = estimates[len(estimates) - 1]
        estimates[estimates < 0] = 0
        return csr_matrix(estimates), estimates.columns

    def process_data(df, res, source, target, verbose, use_raw, **kwargs):
        res['weight'] = res['direction'].apply(apply_weight)
        res.drop_duplicates(subset=['HMDB', 'GENE'], inplace=True)
        final_estimates, metabolites_estimated = estimate_values(df, res, source=source, target=target, weight='weight', verbose=verbose, use_raw=use_raw, **kwargs)
        mask_data = mask[in1d(metabolites, metabolites_estimated), :]
        return final_estimates, metabolites_estimated, mask_data

    if est_fun == 'transport':
        fun = est_fun_dict['ulm']

        df = DataFrame(adata.X.todense(), index=adata.obs_names, columns=adata.var_names)
        final_estimates, metabolites_estimated, mask_pd = process_data(df, me_res, source='HMDB', target='GENE', verbose=verbose, use_raw=False, **kwargs)
        final_estimates_tout, metabolites_estimated_tout, mask_tout = process_data(df, transport_out, source='HMDB', target='GENE', verbose=verbose, use_raw=False, min_n=1, **kwargs)
        final_estimates_tin, metabolites_estimated_tin, mask_tin = process_data(df, transport_in, source='HMDB', target='GENE', verbose=verbose, use_raw=False, min_n=1, **kwargs)

        final_estimates = vstack([final_estimates, final_estimates_tout, final_estimates_tin])
        mask = vstack([mask_pd, mask_tout, mask_tin])
        metabolites_estimated = array([metabolites_estimated, metabolites_estimated_tout, metabolites_estimated_tin])

    else:
        fun = est_fun_dict[est_fun]
        df = DataFrame(adata.X.todense(), index=adata.obs_names, columns=adata.var_names)
        final_estimates, metabolites_estimated, mask = process_data(df, me_res, source='HMDB', target='GENE', verbose=verbose, use_raw=False, **kwargs)

    return final_estimates, mask, metabolites_estimated
