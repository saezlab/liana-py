import numpy as np
from anndata import AnnData
from mudata import MuData
from pandas import DataFrame, concat

from liana._constants import DefaultValues as V
from liana._logging import _check_if_installed
from liana.method._pipe_utils import prep_check_adata
from liana.utils import obsm_to_adata


def estimate_metalinks(adata: AnnData,
                       resource: DataFrame,
                       pd_net: DataFrame,
                       t_net: DataFrame = None,
                       x_name: str = 'metabolite',
                       y_name: str = 'receptor',
                       use_raw: bool = V.use_raw,
                       layer: str = V.layer,
                       verbose: bool = V.verbose,
                       **kwargs) -> MuData:
    """
    Estimate Metabolites from anndata object, and return a MuData object of metabolites and receptors.

    Parameters
    ----------
    adata
        Annotated data matrix.
    resource
        Resource to use for ligand-receptor inference.
    pd_net
        Metabolic production-degradation network to use.
    t_net
        Transport set to use.
    x_name
        Name of the metabolite modality.
    y_name
        Name of the receptor modality. Must be present as a column in the resource.
    %(use_raw)s
    %(layer)s
    %(verbose)s
    **kwargs
        Additional arguments to pass to the decoupler-py functions.
        Method-specific arguments are not supported.

    Returns
    -------
    A MuData object with metabolite & receptor assays.

    """
    dc = _check_if_installed(package_name="decoupler")
    ad = prep_check_adata(adata,
                          layer=layer,
                          use_raw=use_raw,
                          verbose=verbose,
                          groupby=None,
                          min_cells=None,
                          uns=adata.uns,
                          obsm=adata.obsm
                          )
    dc.mt.ulm(ad, net = pd_net, raw=False, verbose=verbose, **kwargs)
    met_est = ad.obsm['score_ulm']

    if t_net is not None:
        dc.mt.waggr(ad, t_net, times=0, raw=False, verbose=verbose, fun='wmean', **kwargs)

        out_est = ad.obsm['score_waggr']
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

    resource = resource[resource['source'].isin(mmat.columns)].copy()
    receptor = ad[:, ad.var.index.isin(np.unique(resource[y_name]))]

    ad.obsm['mmat'] = mmat
    mmat = obsm_to_adata(ad, 'mmat')

    return MuData({x_name:mmat, y_name:receptor})
