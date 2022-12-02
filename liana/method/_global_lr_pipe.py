import numpy as np
import pandas as pd

from liana.resource import select_resource
from liana.method._pipe_utils._reassemble_complexes import explode_complexes
from liana.method._pipe_utils import prep_check_adata, filter_resource, assert_covered, filter_reassemble_complexes
from liana.utils._utils import _get_props


def _global_lr_pipe(adata,
                    resource,
                    resource_name,
                    expr_prop,
                    use_raw,
                    layer,
                    verbose,
                    _key_cols,
                    _complex_cols,
                    _obms_keys,
                    ):
    """

    Parameters
    ----------
    adata

    resource

    resource_name

    expr_prop

    layer

    verbose

    _obms_keys

    _key_cols

    _complex_cols


    Returns
    -------

    """
    # prep adata
    adata = prep_check_adata(adata=adata,
                             use_raw=use_raw,
                             layer=layer,
                             verbose=verbose,
                             groupby=None,
                             min_cells=None,
                             obsm_keys=_obms_keys
                             )

    # select & process resource
    if resource is None:
        resource = select_resource(resource_name.lower())
    resource = explode_complexes(resource)
    resource = filter_resource(resource, adata.var_names)

    # get entities
    entities = np.union1d(np.unique(resource["ligand"]),
                          np.unique(resource["receptor"]))

    # Check overlap between resource and adata  TODO check if this works
    assert_covered(entities, adata.var_names, verbose=verbose)

    # Filter to only include the relevant features
    adata = adata[:, np.intersect1d(entities, adata.var.index)]

    # global_stats
    global_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'props': _get_props(adata.X)},
                                index=adata.var_names).reset_index().rename(
        columns={'index': 'gene'})

    # join global stats to LRs from resource
    lr_res = resource.merge(_rename_means(global_stats, entity='ligand')).merge(
        _rename_means(global_stats, entity='receptor'))

    # get lr_res /w relevant x,y (lig, rec) and filter acc to expr_prop
    lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                         expr_prop=expr_prop,
                                         _key_cols=_key_cols,
                                         complex_cols=_complex_cols
                                         )

    # assign the positions of x, y to the adata
    ligand_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity in
                  lr_res['ligand']}
    receptor_pos = {entity: np.where(adata.var_names == entity)[0][0] for entity in
                    lr_res['receptor']}

    return adata, lr_res, ligand_pos, receptor_pos


def _rename_means(lr_stats, entity):
    df = lr_stats.copy()
    df.columns = df.columns.map(lambda x: entity + '_' + str(x) if x != 'gene' else 'gene')
    return df.rename(columns={'gene': entity})
