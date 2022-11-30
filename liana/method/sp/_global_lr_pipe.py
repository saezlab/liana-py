import numpy as np
import pandas as pd

from liana.resource import select_resource, explode_complexes
from liana.method._pipe_utils import prep_check_adata, filter_resource
from liana.method._liana_pipe import filter_reassemble_complexes


def _global_lr_pipe(adata, resource, resource_name, expr_prop, _obms_keys, _key_cols,
                    _complex_cols):
    """

    Parameters
    ----------
    adata
    resource
    resource_name
    expr_prop
    _obms_keys
    _key_cols
    _complex_cols

    Returns
    -------

    """
    # prep adata
    adata = prep_check_adata(adata, groupby=None, min_cells=None, obsm_keys=_obms_keys)

    # select & process resource
    if resource is None:
        resource = select_resource(resource_name.lower())
    resource = explode_complexes(resource)
    resource = filter_resource(resource, adata.var_names)

    # get entities
    entities = np.union1d(np.unique(resource["ligand"]),
                          np.unique(resource["receptor"])
                          )
    # Filter to only include the relevant features
    adata = adata[:, np.intersect1d(entities, adata.var.index)]

    # global_stats
    global_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'props': adata.X.getnnz(axis=0) / adata.X.shape[0]},
                                index=adata.var_names).reset_index().rename(
        columns={'index': 'gene'})

    # join global stats to LRs from resource
    lr_res = resource.merge(_rename_means(global_stats, entity='ligand')).merge(
        _rename_means(global_stats, entity='receptor'))

    # get lr_res /w relevant x,y (lig, rec) and filter acc to expr_prop
    lr_res = filter_reassemble_complexes(lr_res=lr_res,
                                         _key_cols=_key_cols,
                                         expr_prop=expr_prop,
                                         complex_cols=_complex_cols)

    return lr_res


def _rename_means(lr_stats, entity):
    df = lr_stats.copy()
    df.columns = df.columns.map(lambda x: entity + '_' + str(x) if x != 'gene' else 'gene')
    return df.rename(columns={'gene': entity})
