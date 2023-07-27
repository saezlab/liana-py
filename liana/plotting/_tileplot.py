import numpy as np
import plotnine as p9
from typing import Union, List, Callable, Tuple
import anndata as ad
import pandas as pd

from liana.plotting._common import _prep_liana_res, _get_top_n, _check_var, _filter_by

def tileplot(adata: ad.AnnData = None,
             liana_res: pd.DataFrame = None,
             fill: str = None,
             label: str = None,
             label_fun: Callable = None,
             source_labels: Union[str, List[str]] = None,
             target_labels: Union[str, List[str]] = None,
             ligand_complex: Union[str, List[str]] = None,
             receptor_complex: Union[str, List[str]] = None,
             uns_key: str = 'liana_res',
             top_n: int = None,
             orderby: str = None,
             orderby_ascending: bool = False,
             orderby_absolute: bool = True,
             filterby: str = None,
             filter_lambda: Callable = None,
             figure_size: Tuple[float, float] = (5, 5),
             return_fig: bool = True
             ):
    """
    Tileplot interactions by source and target cells
    
    Parameters
    ----------
    adata
        `AnnData` object with `uns_key` in `adata.uns`. Default is `None`.
    liana_res
        `liana_res` a `DataFrame` in liana's format
    fill
        `column` in `liana_res` to define the fill of the tiles
    label
        `column` in `liana_res` to define the label of the tiles
    label_fun   
        `callable` to apply to the `label` column
    source_labels
        list to specify `source` identities to plot
    target_labels
        list to specify `target` identities to plot
    ligand_complex
        list to specify `ligand_complex` identities to plot
    receptor_complex
        list to specify `receptor_complex` identities to plot
    uns_key
        key in adata.uns where liana_res is stored. Defaults to 'liana_res'.
    top_n
        Obtain only the top_n interactions to plot. Default is `None`
    orderby
        If `top_n` is not `None`, order the interactions by these columns
    orderby_ascending
        If `top_n` is not `None`, specify how to order the interactions
    orderby_absolute
        If `top_n` is not `None`, whether to order by the absolute value of the `orderby` column
    figure_size
        Size of the figure
    return_fig
        `bool` whether to return the fig object, `False` only plots
    
    Returns
    -------
    A `plotnine.ggplot` instance
    
    """    
    liana_res = _prep_liana_res(adata=adata,
                                liana_res=liana_res,
                                source_labels=source_labels,
                                target_labels=target_labels,
                                ligand_complex=ligand_complex,
                                receptor_complex=receptor_complex,
                                uns_key=uns_key)
    
    liana_res = _filter_by(liana_res, filterby, filter_lambda)
    liana_res = _get_top_n(liana_res, top_n, orderby, orderby_ascending, orderby_absolute)
    
    # get columns which ends with fill or label
    relevant_cols = [col for col in liana_res.columns if col.endswith(fill) | col.endswith(label)]
    
    ligand_stats = _entity_stats(liana_res,
                                 entity='ligand',
                                 entity_type='source',
                                 relevant_cols=relevant_cols)
    
    _check_var(ligand_stats, var=fill, var_name='fill')
    _check_var(ligand_stats, var=label, var_name='label')

    receptor_stats = _entity_stats(liana_res,
                                   entity='receptor',
                                   entity_type='target',
                                   relevant_cols=relevant_cols)

    liana_res = pd.concat([ligand_stats, receptor_stats])

    if label_fun is not None:
        liana_res[label] = liana_res[label].apply(label_fun)
        
    p = (
        p9.ggplot(liana_res, p9.aes(x='cell_type', y='interaction', fill=fill)) +
        p9.geom_tile() +
        p9.geom_text(p9.aes(label=label), size=10, color='white') +
        p9.facet_grid(facets='~ type', scales='free') +
        p9.theme_bw(base_size=14) +
        p9.theme(
            axis_text_x=p9.element_text(angle=90),
            figure_size=figure_size,
            strip_background=p9.element_rect(colour="black", fill="#fdfff4"),
        ) +
        p9.labs(x='Cell type', y='Interaction', fill=str.capitalize(fill))
    )
    
    if return_fig:
        return p
    
    p.draw()
    
def _entity_stats(liana_res, entity, entity_type, relevant_cols):
    entity_stats = liana_res[['interaction', f"{entity}_complex", entity_type, *relevant_cols]].copy()
    entity_stats = entity_stats.rename(columns={entity_type: 'cell_type'}).assign(type=entity_type.capitalize())
    entity_stats.columns = entity_stats.columns.str.replace(entity + '_', '')
    return entity_stats