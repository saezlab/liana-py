import numpy as np
import pandas as pd
from anndata import AnnData

from types import ModuleType
from liana.method._pipe_utils._common import _get_props

from liana.method.sp._misty._Misty import MistyData

from liana.utils.spatial_neighbors import spatial_neighbors
from liana.method._pipe_utils._pre import _choose_mtx_rep

from liana.resource import select_resource
from liana.method._pipe_utils import prep_check_adata
from liana.method.sp._utils import _add_complexes_to_var
from liana._logging import _check_if_installed

def _make_view(adata, nz_threshold=0.1, add_obs=False, use_raw=False,
               layer=None, connecitivity=None, spatial_key=None, verbose=False):

    X = _choose_mtx_rep(adata=adata, use_raw=use_raw, layer=layer, verbose=verbose)

    obsm = dict()
    obsp = dict()
    if spatial_key is not None:
        if spatial_key not in adata.obsm.keys():
            raise ValueError(f"spatial_key {spatial_key} not found in `obsm`")
        obsm[spatial_key] = adata.obsm[spatial_key]

        if connecitivity is not None:
            obsp = dict()
            obsp[f"{spatial_key}_connectivities"] = connecitivity

    obs = adata.obs if add_obs else pd.DataFrame(index=adata.obs.index)

    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=adata.var_names),
                    obsp=obsp, obsm=obsm, dtype=np.float32)
    var_msk = _get_props(adata.X) >= nz_threshold
    adata = adata[:, var_msk]

    return adata


def genericMistyData(intra,
                     intra_use_raw=False,
                     intra_layer=None,
                     extra=None,
                     extra_use_raw=False,
                     extra_layer=None,
                     nz_threshold=0.1,
                     add_para=True,
                     spatial_key='spatial',
                     set_diag=False,
                     kernel = 'misty_rbf', ## TODO change to gaussian kernel
                     bandwidth = 100,
                     zoi = 0,
                     cutoff = 0.1,
                     add_juxta=True,
                     n_neighs = 6,
                     verbose=False,
                     **kwargs,
                     ):

    """
    Construct a MistyData object from an AnnData object with views as presented in the manuscript.

    Parameters
    ----------
    intra : `anndata.AnnData`
        AnnData object with the intraview
    intra_use_raw : `bool`, optional (default: False)
        Whether to use the raw data of the intraview.
    intra_layer : `str`, optional (default: None)
        The layer of the intraview to use.
    extra : `anndata.AnnData`, optional (default: None)
        AnnData object with the extraview(s). If None, the extraview is set to be the same as the intraview.
    extra_use_raw : `bool`, optional (default: False)
        Whether to use the raw data of the extraview.
    extra_layer : `str`, optional (default: None)
        The layer of the extraview(s) to use.
    nz_threshold: `float`, optional (default: 0.1)
        The threshold for the number of non-zero entries in each view.
    add_para : `bool`, optional (default: True)
        Whether to add the paraview.
    spatial_key : `str`, optional (default: 'spatial')
        The key in adata.obsm where the spatial coordinates are stored.
    set_diag : `bool`, optional (default: True)
        Whether to set the diagonal of the connectivity matrix to 1.
    kernel : `str`, optional (default: 'misty_rbf')
        A radial basis function kernel to use for the generation of the connectivity matrix for the paraview.
        Default is 'misty_rbf', a kernel derivative of a Gaussian kernel.
    bandwidth : `float`, optional (default: 100)
        The bandwidth of the kernel.
    zoi : `float`, optional (default: 0)
        The zone of indifference of the kernel, i.e. the kernel is set to 0 for distances smaller than zoi.
    cutoff : `float`, optional (default: 0.1)
        The cutoff for the connectivity matrix.
    add_juxta : `bool`, optional (default: True)
        Whether to add the juxtaview. The juxtaview is constructed using `squidpy.gr.spatial_neighbors`,
        and should represent the direct spatial neighbors of each cell/spot.
    n_neighs : `int`, optional (default: 6)
        The number of neighbors to consider when constructing the juxtaview.
    verbose : `bool`, optional (default: False)
        Whether to print progress.
    **kwargs : `dict`, optional
        Additional arguments to pass to `squidpy.gr.spatial_neighbors`.

    Returns
    -------
    `MistyData` object with the intra view, and two fixed extra view(s): para and juxta.

    """
    # init views
    views = {}
    intra = _make_view(adata=intra, nz_threshold=nz_threshold, add_obs=True,
                       use_raw=intra_use_raw, layer=intra_layer,
                       spatial_key=spatial_key, verbose=verbose)
    views['intra'] = intra

    if extra is None:
        extra = intra

    if add_juxta:
        sq = _check_if_installed('squidpy')
        neighbors, _ = sq.gr.spatial_neighbors(adata=extra,
                                               copy=True,
                                               spatial_key=spatial_key,
                                               set_diag=set_diag,
                                               n_neighs=n_neighs,
                                               **kwargs
                                               )

        views['juxta'] = _make_view(adata=extra, nz_threshold=nz_threshold,
                                    use_raw=extra_use_raw, layer=extra_layer,
                                    spatial_key=spatial_key, connecitivity=neighbors,
                                    verbose=verbose)

    if add_para:
        weights = spatial_neighbors(adata=extra,
                                    spatial_key=spatial_key,
                                    bandwidth=bandwidth,
                                    kernel=kernel,
                                    set_diag=set_diag,
                                    inplace=False,
                                    cutoff=cutoff,
                                    zoi=zoi
                                    )
        views['para'] = _make_view(adata=extra, nz_threshold=nz_threshold,
                                   use_raw=extra_use_raw, layer=extra_layer,
                                   spatial_key=spatial_key, connecitivity=weights,
                                   verbose=verbose)

    return MistyData(views, intra.obs, spatial_key)


def _check_if_squidpy() -> ModuleType:
    try:
        import squidpy as sq
    except Exception:

        raise ImportError(
            'squidpy is not installed. Please install it with: '
            'pip install squidpy'
        )
    return sq


def lrMistyData(adata,
                resource_name='consensus',
                resource=None,
                nz_threshold=0.1,
                use_raw = False,
                layer = None,
                spatial_key='spatial', ## TODO Change to Gaussian kernel
                kernel = 'misty_rbf',
                bandwidth = 100,
                set_diag = False,
                cutoff = 0.1,
                zoi = 0,
                verbose = False
                ):
    """
    Generate a MistyData object from an AnnData object in ligand-receptor format.

    Parameters
    ----------
    adata : `anndata.AnnData`
        AnnData object
    resource_name : `str`, optional (default: 'consensus')
        The name of the resource to use. See `show_resources` for available resources.
    resource : `pandas.DataFrame`, optional (default: None)
        A resource in the form of a pandas DataFrame. If None, the resource is selected using `select_resource`.
    nz_threshold : `float`, optional (default: 0.1)
        The threshold for the number of non-zero entries in each view.
    use_raw : `bool`, optional (default: False)
        Whether to use the raw data of the AnnData object.
    layer : `str`, optional (default: None)
        The layer of the AnnData object to use.
    spatial_key : `str`, optional (default: 'spatial')
        The key in adata.obsm where the spatial coordinates are stored.
    kernel : `str`, optional (default: 'misty_rbf')
        A radial basis function kernel to use for the generation of the connectivity matrix for the extra view.
        Default is 'misty_rbf', a kernel derivative of a Gaussian kernel.
    bandwidth : `float`, optional (default: 100)
        The bandwidth of the kernel.
    set_diag : `bool`, optional (default: True)
        Whether to set the diagonal of the connectivity matrix to 1.
    cutoff : `float`, optional (default: 0.1)
        The minimum value cutoff for the connectivity matrix.
    zoi : `float`, optional (default: 0)
        Zone of indifference of the kernel, i.e. the kernel is set to 0 for distances smaller than zoi.
    verbose : `bool`, optional (default: False)
        Whether to print progress.

    Returns
    -------
    A `MistyData` object with receptors in the intra view & ligands in the extra view.
    """
    # TODO: reduce redundancies in documentation
    if resource is None:
        resource = select_resource(resource_name)

    adata = prep_check_adata(adata=adata,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose,
                             groupby=None,
                             min_cells=None,
                             obsm = {spatial_key: adata.obsm[spatial_key]}
                             )

    adata = _add_complexes_to_var(adata,
                                  np.union1d(resource['receptor'].astype(str),
                                             resource['ligand'].astype(str)
                                             )
                                  )

    # filter_resource after adding complexes to var
    resource = resource[(np.isin(resource.ligand, adata.var_names)) &
                        (np.isin(resource.receptor, adata.var_names))]

    views = dict()
    views['intra'] =  _make_view(adata=adata[:, resource['receptor'].unique()],
                        nz_threshold=0, add_obs=True)

    connectivity = spatial_neighbors(adata=adata, spatial_key=spatial_key,
                                     bandwidth=bandwidth, kernel=kernel,
                                     set_diag=set_diag, cutoff=cutoff,
                                     zoi=zoi, inplace=False)

    views['extra'] = _make_view(adata=adata[:,resource['ligand'].unique()],
                                spatial_key=spatial_key, nz_threshold=nz_threshold,
                                connecitivity=connectivity)

    return MistyData(data=views, obs=views['intra'].obs, spatial_key=spatial_key)
