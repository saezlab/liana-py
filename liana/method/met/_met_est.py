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
        met_net = _get_met_sets() # put functions to resources later
    
    fun(adata, net = met_net, use_raw = False, source = 'HMDB', target = 'Symbol', weight = 'Direction')
    met_est = adata.obsm['ulm_estimate'] #problem with how to call, maybe you now better solution
    #met_est = obsm_to_adata(met_est, 'metabolite_estimate') 

    if transport_sets is None:
        t_in_net, t_out_net = _transport_sets() # put functions to resources later
    else:
        t_in_net, t_out_net = _split_transport(transport_sets)
    
    fun(adata, t_in_net,  use_raw = False, source = 'HMDB', target = 'Symbol', weight = 'Direction')
    in_est = adata.obsm['ulm_estimate']

    fun(adata, t_out_net,  use_raw = False, source = 'HMDB', target = 'Symbol', weight = 'Direction')
    out_est = adata.obsm['ulm_estimate']

    in_mask = in_est > 0
    out_mask = out_est > 0

    t_mask = out_mask #* in_mask
    mask = DataFrame(1, index = met_est.index, columns = met_est.columns)
    mask[t_mask == 0] = 0

    receptor_expr = adata[:, adata.var.index.isin(resource['receptor'])]

    mmat = met_est * mask
    mmat[mmat < 0] = 0
    mmat = AnnData(mmat)
    mdata = MuData({'metabolite':mmat, 'rna':receptor_expr})

    return mdata


def _get_met_sets():
    met_sets= read_csv("/home/efarr/Documents/GitHub/metalinks/metalinksDB/PD_20230802.csv")
    met_net = met_sets[met_sets['Reversibility'] != 'reversible']
    met_net = met_net[~met_net['T_direction'].isin(['out', 'in'])]
    met_net['Direction'].replace({'degrading': -1, 'producing': 1}, inplace=True)
    met_net['Direction'] = met_net['Direction'].astype(int)

    return met_net

def _transport_sets(): # can be made more beautiful
    met_sets = read_csv("/home/efarr/Documents/GitHub/metalinks/metalinksDB/PD_20230802.csv")
    t_net = met_sets[met_sets['T_direction'].isin(['out', 'in'])]
    t_net['Direction'] = where(t_net['Reversibility'] == 'reversible', 1, t_net['Direction'])

    t_in = t_net.copy()  # Make a copy to avoid modifying the original DataFrame
    t_in.loc[(t_in['T_direction'] == 'in') & (t_in['Reversibility'] == 'irreversible'), 'Direction'] = 1
    t_in.loc[(t_in['T_direction'] == 'out') & (t_in['Reversibility'] == 'irreversible'), 'Direction'] = -1
    t_in['Direction'] = t_in['Direction'].astype(int)

    t_out = t_net.copy()  # Make a copy to avoid modifying the original DataFrame
    t_out.loc[(t_out['T_direction'] == 'out') & (t_out['Reversibility'] == 'irreversible'), 'Direction'] = 1
    t_out.loc[(t_out['T_direction'] == 'in') & (t_out['Reversibility'] == 'irreversible'), 'Direction'] = -1
    t_out['Direction'] = t_out['Direction'].astype(int)


    return t_in, t_out


def _split_transport(transport_sets):
    t_in_net = transport_sets.copy()
    t_out_net = transport_sets.copy()
    t_in_net = t_in_net[t_in_net['direction'] == 'in']
    t_out_net = t_out_net[t_out_net['direction'] == 'out']
    return t_in_net, t_out_net