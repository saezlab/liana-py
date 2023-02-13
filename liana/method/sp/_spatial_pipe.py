import numpy as np
import pandas as pd
from itertools import product

from liana.resource import select_resource
from liana.method._pipe_utils._reassemble_complexes import explode_complexes
from liana.method._pipe_utils import prep_check_adata, filter_resource, assert_covered, filter_reassemble_complexes
from liana.utils._utils import _get_props



def global_bivariate_pipe(mdata, x_mod, y_mod, nz_threshold=0):
    """
    Global Bivariate analysis pipeline
    
    Parameters
    ----------
    mdata : anndata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and
    x_mod : str
        Name of the modality to use as x
    y_mod : str
        Name of the modality to use as y
    nz_threshold : float
        Threshold for the proportion of non-zero values in a cell for the interaction to be considered
        
    Returns
    -------
    Returns xy_stats, x_pos, y_pos
    
    """
    
    xdata = mdata[x_mod] ## TODO takes both anndatas and strings for mudata modalities
    ydata = mdata[y_mod]
    
    x_stats = _rename_means(_anndata_to_stats(xdata, nz_threshold), entity='x')
    y_stats = _rename_means(_anndata_to_stats(ydata, nz_threshold), entity='y')
    
    xy_stats = pd.DataFrame(list(product(x_stats['x_entity'], 
                                             y_stats['y_entity'])
                                 ),
                            columns=['x_entity', 'y_entity'])
    
    # join global stats to LRs from resource
    xy_stats = xy_stats.merge(x_stats).merge(y_stats)
    
    xy_stats['interaction'] = xy_stats['x_entity'] + '&' + xy_stats['y_entity']
    
    # Remove self-interactions (e.g. x = x) # TODO: check if this is necessary, pointless for bivariate metrics
    xy_stats = xy_stats[xy_stats['x_entity'] != xy_stats['y_entity']]
    
    # assign the positions of x, y to the adata
    x_pos = {entity: np.where(xdata.var_names == entity)[0][0] for entity in xy_stats['x_entity']}
    y_pos = {entity: np.where(ydata.var_names == entity)[0][0] for entity in xy_stats['y_entity']}
    # reorder columns
    xy_stats = xy_stats.reindex(columns=sorted(xy_stats.columns))
    
    return xy_stats, x_pos, y_pos





def _anndata_to_stats(adata, nz_thr=0.1):
    from scipy.sparse import csr_matrix
    adata.X = csr_matrix(adata.X) ## TODO change to ~prep_check_adata (but not for gene expression alone)
    
    global_stats = pd.DataFrame({'means': adata.X.mean(axis=0).A.flatten(),
                                 'non_zero': _get_props(adata.X)},
                                index=adata.var_names)
    global_stats = global_stats.reset_index().rename(columns={'index': 'entity'})
    global_stats = global_stats[global_stats['non_zero'] >= nz_thr]

    return global_stats





### TODO into another file & complete pipe (as liana in sc)
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
    Global Spatial ligand-receptor analysis pipeline

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and
    resource
        Ligand-receptor resource as a pandas dataframe with columns: ligand, receptor
    resource_name
        Name of the resource
    expr_prop
        Proportion of expression of the ligand and receptor in a cell for the interaction to be considered
    layer
        Layer to use for the analysis. If None, use raw or X
    verbose
        Verbosity
    _obms_keys
        Keys of the adata.obsm to use for the analysis
    _key_cols
        Columns to use as keys for the resource
    _complex_cols
        Columns to use as keys for the complexes

    Returns
    -------
    Returns adata, lr_res, ligand_pos, receptor_pos
    

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


def _get_ordered_matrix(mat, pos, order):
    _indx = np.array([pos[x] for x in order])
    return mat[:, _indx].T


