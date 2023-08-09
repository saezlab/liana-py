from muon import MuData
import numpy as np
from pandas import read_csv, DataFrame
from liana.utils import obsm_to_adata


def estimate_metalinks(adata, resource, fun=None, met_net=None, transport_sets=None, consider_transport = True, **kwargs):
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
    
    net = met_net[met_net['Type'] == 'met_est']
    fun(adata, net = net, use_raw = False, source = 'HMDB', target = 'Symbol', weight = 'Direction', **kwargs)
    met_est = adata.obsm['ulm_estimate']

    if consider_transport:

        if transport_sets is None:
            net = met_net[met_net['Type'] == 'export']
        else:
            net = transport_sets

        fun(adata, net,  use_raw = False, source = 'HMDB', target = 'Symbol', weight = 'Direction', **kwargs)
        out_est = adata.obsm['ulm_estimate']

        out_mask = out_est > 0

        mask = np.ones(out_est.shape)
        mask[out_mask == 0] = 0

        mmat = met_est * mask
       
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

def _get_met_sets(type):
    return read_csv("liana/resource/PD_processed.csv")
