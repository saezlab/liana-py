import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.interpolate import griddata
from anndata import AnnData

from liana.method._pipe_utils._pre import _choose_mtx_rep
from liana._docs import d
from liana._constants import DefaultValues as V


@d.dedent
def interpolate_adata(target: AnnData,
                      reference: AnnData,
                      spatial_key: str,
                      layer=V.layer,
                      use_raw=V.use_raw,
                      method: str = 'linear',
                      fill_value: float = 0,
                      verbose=V.verbose) -> AnnData:
    """
    Interpolates spatial data from a target AnnData object to a reference AnnData object based on spatial coordinates.
    The function creates a new AnnData object where the `.X` attribute is filled with interpolated data using the specified method.

    Parameters
    -----------
    target
        The AnnData object to be interpolated.
    reference
        The AnnData object to be used as reference.
    %(spatial_key)s
    %(layer)s
    %(use_raw)s
    method
        Interpolation method. See `scipy.interpolate.griddata` for more information.
    fill_value
        Value to fill in for points outside of the convex hull of the input points.
    %(verbose)s

    Returns
    --------
    AnnData: A new AnnData object with the same metadata as the reference but with interpolated spatial data in `.X`.
    """
    target_coords = target.obsm[spatial_key]
    reference_coords = reference.obsm[spatial_key]

    X = lil_matrix((reference.shape[0], target.shape[1]), dtype=np.float32)

    ad = AnnData(X=X,
                 uns=reference.uns,
                 obs=reference.obs,
                 obsm=reference.obsm,
                 obsp=reference.obsp,
                 var=target.var,
                 varm=target.varm
                 )

    values = _choose_mtx_rep(adata=target, use_raw=use_raw, layer=layer, verbose=verbose).A

    ad.X = csr_matrix(
        griddata(points=target_coords,
                 xi=reference_coords,
                 values=values,
                 method=method,
                 fill_value=fill_value
                 )
        )

    return ad
