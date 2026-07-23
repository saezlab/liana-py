from types import ModuleType

import numpy as np
import pandas as pd
from anndata import AnnData

from liana.method._pipe_utils import prep_check_adata
from liana.method._pipe_utils._common import _get_props
from liana.method._pipe_utils._pre import _choose_mtx_rep
from liana.method.sp._misty._Misty import MistyData
from liana.method.sp._utils import _add_complexes_to_var
from liana.resource import select_resource
from liana.utils.spatial_neighbors import spatial_neighbors


def _make_view(adata, nz_threshold=0.1, add_obs=False, use_raw=False,
               layer=None, connecitivity=None, spatial_key=None, verbose=False):

    X = _choose_mtx_rep(adata=adata, use_raw=use_raw, layer=layer, verbose=verbose)

    obsm = {}
    obsp = {}
    if spatial_key is not None:
        if spatial_key not in adata.obsm.keys():
            raise ValueError(f"spatial_key {spatial_key} not found in `obsm`")
        obsm[spatial_key] = adata.obsm[spatial_key]

        if connecitivity is not None:
            obsp = {}
            obsp[f"{spatial_key}_connectivities"] = connecitivity

    obs = adata.obs if add_obs else pd.DataFrame(index=adata.obs.index)

    adata = AnnData(X=X.astype(np.float32, copy=False), obs=obs, var=pd.DataFrame(index=adata.var_names),
                    obsp=obsp, obsm=obsm)
    var_msk = _get_props(adata.X) >= nz_threshold
    adata = adata[:, var_msk]

    return adata


def genericMistyData(intra: AnnData,
                     intra_use_raw: bool = False,
                     intra_layer: str = None,
                     extra: AnnData = None,
                     extra_use_raw: bool = False,
                     extra_layer: str = None,
                     nz_threshold: float = 0.1,
                     add_para: bool = True,
                     spatial_key: str = 'spatial',
                     set_diag: bool = False,
                     kernel: str = 'misty_rbf',
                     bandwidth: float = 100,
                     zoi: float = 0,
                     cutoff: float = 0.1,
                     add_juxta: bool = True,
                     n_neighs: int = 6,
                     max_neighs: int = 18,
                     verbose: bool = False
                     ) -> MistyData:
    """
    Construct a MistyData object from an AnnData object with views as presented in the manuscript.

    Parameters
    ----------
    intra
        AnnData object with the intraview
    intra_use_raw
        Whether to use the raw data of the intraview.
    intra_layer
        The layer of the intraview to use.
    extra
        AnnData object with the extraview(s). If None, the extraview is set to be the same as the intraview.
    extra_use_raw
        Whether to use the raw data of the extraview.
    extra_layer
        The layer of the extraview(s) to use.
    nz_threshold
        The threshold for the number of non-zero entries in each view.
    add_para
        Whether to add the paraview.
    spatial_key
        The key in adata.obsm where the spatial coordinates are stored.
    set_diag
        Whether to set the diagonal of the connectivity matrix to 1.
    kernel
        A radial basis function kernel to use for the generation of the connectivity matrix for the paraview.
        Default is 'misty_rbf', a kernel derivative of a Gaussian kernel.
    bandwidth
        The bandwidth of the kernel.
    zoi
        The zone of indifference of the kernel, i.e. the kernel is set to 0 for distances smaller than zoi.
    cutoff
        The cutoff for the connectivity matrix.
    add_juxta
        Whether to add the juxtaview. The juxtaview is constructed using only the nearest neighbors.
        A bandwidth of 5 times the bandwidth of the paraview is used to ensure that the nearest neighbors within the radius.
    n_neighs
        The number of neighbors to consider when constructing the juxtaview.
    max_neighs
        The maximum number of neighbors to consider when constructing the Paraview.
    verbose
        Whether to print progress.

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
        neighbors = spatial_neighbors(extra,
                                      bandwidth=bandwidth*5,
                                      spatial_key=spatial_key,
                                      max_neighbours=n_neighs,
                                      set_diag=set_diag,
                                      inplace=False)

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
                                    max_neighbours=max_neighs,
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
    except ImportError:
        raise ImportError(
            'squidpy is not installed. Please install it with: '
            'pip install squidpy'
        ) from None
    return sq


def lrMistyData(adata: AnnData,
                resource_name: str = 'consensus',
                resource: pd.DataFrame = None,
                nz_threshold: float = 0.1,
                use_raw: bool = False,
                layer: str = None,
                spatial_key: str = 'spatial',
                kernel: str = 'misty_rbf',
                bandwidth: float = 100,
                set_diag: bool = False,
                cutoff: float = 0.1,
                zoi: float = 0,
                verbose: bool = False
                ) -> MistyData:
    """
    Generate a MistyData object from an AnnData object in ligand-receptor format.

    Parameters
    ----------
    adata
        AnnData object
    resource_name
        The name of the resource to use. See `show_resources` for available resources.
    resource
        A resource in the form of a pandas DataFrame. If None, the resource is selected using `select_resource`.
    nz_threshold
        The threshold for the number of non-zero entries in each view.
    use_raw
        Whether to use the raw data of the AnnData object.
    layer
        The layer of the AnnData object to use.
    spatial_key
        The key in adata.obsm where the spatial coordinates are stored.
    kernel
        A radial basis function kernel to use for the generation of the connectivity matrix for the extra view.
        Default is 'misty_rbf', a kernel derivative of a Gaussian kernel.
    bandwidth
        The bandwidth of the kernel.
    set_diag
        Whether to set the diagonal of the connectivity matrix to 1.
    cutoff
        The minimum value cutoff for the connectivity matrix.
    zoi
        Zone of indifference of the kernel, i.e. the kernel is set to 0 for distances smaller than zoi.
    verbose
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

    views = {}
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


def lrMistyDataByCellType(adata: AnnData,
                          resource_name: str = 'consensus',
                          resource: pd.DataFrame = None,
                          nz_threshold: float = 0.1,
                          use_raw: bool = False,
                          layer: str = None,
                          spatial_key: str = 'spatial',
                          celltype_key: str = 'cell_type',
                          kernel: str = 'misty_rbf',
                          bandwidth: float = 100,
                          set_diag: bool = False,
                          cutoff: float = 0.1,
                          zoi: float = 0,
                          verbose: bool = False
                          ) -> MistyData:
    """
    Generate a MistyData object from an AnnData object in ligand-receptor format.

    Parameters
    ----------
    adata
        AnnData object
    resource_name
        The name of the resource to use. See `show_resources` for available resources.
    resource
        A resource in the form of a pandas DataFrame. If None, the resource is selected using `select_resource`.
    nz_threshold
        The threshold for the number of non-zero entries in each view.
    use_raw
        Whether to use the raw data of the AnnData object.
    layer
        The layer of the AnnData object to use.
    spatial_key
        The key in adata.obsm where the spatial coordinates are stored.
    celltype_key
        The key in adata.obs where the cell type labels are stored.
    kernel
        A radial basis function kernel to use for the generation of the connectivity matrix for the extra view.
        Default is 'misty_rbf', a kernel derivative of a Gaussian kernel.
    bandwidth
        The bandwidth of the kernel.
    set_diag
        Whether to set the diagonal of the connectivity matrix to 1.
    cutoff
        The minimum value cutoff for the connectivity matrix.
    zoi
        Zone of indifference of the kernel, i.e. the kernel is set to 0 for distances smaller than zoi.
    verbose
        Whether to print progress.

    Returns
    -------
    A `MistyData` object with receptors in the intra view and ligands in an
    extra view for each cell type. Receiver groups are selected during fitting
    with categorical ``maskby``.
    """
    if resource is None:
        resource = select_resource(resource_name)

    if celltype_key not in adata.obs:
      raise KeyError(
          f"'{celltype_key}' not found in adata.obs."
        )

    adata = prep_check_adata(adata=adata,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose,
                             groupby=None,
                             min_cells=None,
                             obsm = {spatial_key: adata.obsm[spatial_key]}
                             )

    celltypes = (
        adata.obs[celltype_key]
        .astype("category")
        .cat.remove_unused_categories()
    )

    celltype_names = celltypes.cat.categories.tolist()

    adata = _add_complexes_to_var(adata,
                                  np.union1d(resource['receptor'].astype(str),
                                             resource['ligand'].astype(str))
                                  )

    resource = resource[(np.isin(resource.ligand, adata.var_names)) &
                        (np.isin(resource.receptor, adata.var_names))]

    receptors = resource["receptor"].drop_duplicates().to_numpy()
    ligands = resource["ligand"].drop_duplicates().to_numpy()

    sender_celltypes = celltype_names

    views = {}

    # Intra
    views["intra"] = _make_view(
        adata=adata[:, receptors],
        spatial_key=spatial_key,
        nz_threshold=0,
        add_obs=True,
    )

    # Compute the spatial connectivity matrix once and reuse it for all
    # sender-specific extra views.
    connectivity = spatial_neighbors(adata=adata, spatial_key=spatial_key,
                                     bandwidth=bandwidth, kernel=kernel,
                                     set_diag=set_diag, cutoff=cutoff,
                                     zoi=zoi, inplace=False)

    # Construct one ligand view for each non-receiver cell type.

    # NOTE: Should the celltype from the intraview also get an extraview?
    for celltype in sender_celltypes:
        sender_mask = (celltypes == celltype).to_numpy()

        ligand_adata = adata[:, ligands].copy()
        ligand_adata.X = ligand_adata.X.multiply(
            sender_mask[:, None]
        ).tocsr()
        ligand_adata.X.eliminate_zeros()

        views[f"extra_{celltype}"] = _make_view(
            adata=ligand_adata,
            spatial_key=spatial_key,
            nz_threshold=nz_threshold,
            connecitivity=connectivity,
        )

    misty = MistyData(
        data=views,
        obs=views["intra"].obs,
        spatial_key=spatial_key,
    )
    misty.uns["_misty_ligands_by_receptor"] = (
        resource.groupby("receptor")["ligand"]
        .agg(list)
        .to_dict()
    )
    misty.uns["_misty_by_cell_type"] = True
    misty.uns["_misty_celltype_key"] = celltype_key
    return misty
