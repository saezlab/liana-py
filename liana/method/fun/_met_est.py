from muon import MuData
import decoupler as dc
import numpy as np
from pandas import read_csv, DataFrame, concat
from liana.utils import obsm_to_adata


def estimate_metalinks(adata, resource, met_net=None, consider_transport = True, **kwargs):
    """
    Estimate metalinks from anndata object.

    Parameters
    ----------
    adata
        Annotated data matrix.
    fun
        decoupler-py function to use.
    met_net
        Metabolic network to use.
    transport_set
        Transport set to use.
    kwargs
        Additional arguments for the decoupling function.

    Returns
    -------
    A MuData object with metabolite & receptor assays.

    """

    if met_net is None:
        met_net = _get_met_sets() # needs configuration with get_resource .. 
    met_net = met_net[met_net['HMDB'].isin(np.unique(resource['ligand']))]
    
    net = met_net[met_net['Type'] == 'met_est'].drop_duplicates(['HMDB', 'Symbol']).copy()
    
    dc.run_ulm(adata, net = net, use_raw = False, source = 'HMDB', target = 'Symbol', weight = 'Direction', **kwargs)
    met_est = adata.obsm['ulm_estimate']

    if consider_transport:

        net = met_net[met_net['Type'] == 'export'].drop_duplicates(['HMDB', 'Symbol', 'Direction'])

        dc.run_wmean(adata, net, source = 'HMDB', target = 'Symbol', weight = 'Direction', times=0, min_n=3)
        
        out_est = adata.obsm['wmean_estimate']
        intersect = np.intersect1d(met_est.columns, out_est.columns)
        out_est = out_est[intersect]

        out_mask = out_est > 0

        mask = np.ones(out_est.shape)
        mask[out_mask == 0] = 0

        # mask those with transporters
        mmat = met_est[intersect] * mask
        
        # concat the rest
        coldiff = np.setdiff1d(met_est.columns, mmat.columns)
        mmat = concat([mmat, met_est[coldiff]], axis = 1)
       
    else:
        mmat = met_est

    mmat[mmat < 0] = 0
    receptor_expr = adata[:, adata.var.index.isin(resource['receptor'])]

    adata.obsm['mmat'] = mmat
    mmat = obsm_to_adata(adata, 'mmat')
    
    mdata = MuData({'metabolite':mmat, 'rna':receptor_expr})
    
    mdata.obsp = adata.obsp
    mdata.uns = adata.uns
    mdata.obsm = adata.obsm

    return mdata

def _get_met_sets():
    return read_csv("liana/resource/PD_processed.csv")
