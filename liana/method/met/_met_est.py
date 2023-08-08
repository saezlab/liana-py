# Dont name it metalinks on purpose
from muon import MuData
import decoupler as dc
from pandas import read_csv, DataFrame
from liana.utils import obsm_to_adata
from numpy import where
from scanpy import AnnData


def estimate_metalinks(adata, resource, fun=dc.run_ulm, met_net=None, transport_sets=None, **kwargs):
    """
    Estimate metalinks from anndata object.

    Parameters
    ----------
    adata
        Annotated data matrix.
    dc.fun
        Decoupling function to use.
    met_net
        Metabolic network to use.
    transport_set
        Transport set to use.
    kwargs
        Additional arguments for the decoupling function.

    Returns
    -------
    A MuData object with ``adata.metalinks.mmat`` and ``adata.metalinks.rmat``.

    """

    if met_net is None:
        met_net = _get_met_sets() # needs configuration with get_resource .. 
    
    fun(adata, net = met_net, use_raw = False, source = 'HMDB', target = 'Symbol', weight = 'Direction')
    met_est = adata.obsm['ulm_estimate']

    if transport_sets is None:
        t_out_net = _transport_sets()
    else:
        t_out_net = transport_sets

    fun(adata, t_out_net,  use_raw = False, source = 'HMDB', target = 'Symbol', weight = 'Direction')
    out_est = adata.obsm['ulm_estimate']

    out_mask = out_est > 0

    t_mask = out_mask # connection with import ? 
    mask = DataFrame(1, index = met_est.index, columns = met_est.columns)
    mask[t_mask == 0] = 0

    receptor_expr = adata[:, adata.var.index.isin(resource['receptor'])]

    mmat = met_est * mask
    mmat[mmat < 0] = 0
    adata.obsm['mmat'] = mmat
    mmat = obsm_to_adata(adata, 'mmat')
    mdata = MuData({'metabolite':mmat, 'rna':receptor_expr})

    return mdata

def _get_met_sets():
    met_sets= read_csv("liana/resource/PD_processed.csv")
    met_net = met_sets[met_sets['Type'] == 'met_est']

    return met_net

def _transport_sets(): 
    met_sets = read_csv("liana/resource/PD_processed.csv")
    t_out = met_sets[met_sets['Type'] == 'export']
    return t_out
