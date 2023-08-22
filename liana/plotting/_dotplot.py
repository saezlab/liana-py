from __future__ import annotations

import anndata
import pandas

from plotnine import ggplot, geom_point, aes, \
    facet_grid, labs, theme_bw, theme, element_text, element_rect, scale_size_continuous, scale_color_cmap
    
from liana.plotting._common import _prep_liana_res, _check_var, _get_top_n, _filter_by, _inverse_scores


def dotplot(adata: anndata.AnnData = None,
            uns_key = 'liana_res',
            liana_res: pandas.DataFrame = None,
            colour: str = None,
            size: str = None,
            source_labels: list = None,
            target_labels: list = None,
            top_n: int = None,
            orderby: str | None = None,
            orderby_ascending: bool | None = None,
            orderby_absolute: bool = False,
            filterby: bool | None = None,
            filter_lambda=None,
            ligand_complex: str | None = None, 
            receptor_complex: str | None = None,
            inverse_colour: bool = False,
            inverse_size: bool = False,
            cmap: str = 'viridis',
            size_range: tuple = (2, 9),
            figure_size: tuple = (8, 6),
            return_fig=True) -> ggplot:
    """
    Dotplot interactions by source and target cells

    Parameters
    ----------
    adata
        `AnnData` object with `uns_key` in `adata.uns`. Default is `None`.
    uns_key
        key in adata.uns where liana_res is stored. Defaults to 'liana_res'.
    liana_res
        `liana_res` a `DataFrame` in liana's format
    colour
        `column` in `liana_res` to define the colours of the dots
    size
        `column` in `liana_res` to define the size of the dots
    source_labels
        list to specify `source` identities to plot
    target_labels
        list to specify `target` identities to plot
    top_n
        Obtain only the top_n interactions to plot. Default is `None`
    orderby
        If `top_n` is not `None`, order the interactions by these columns
    orderby_ascending
        If `top_n` is not `None`, specify how to order the interactions
    orderby_absolute
        If `top_n` is not `None`, whether to order by the absolute value of the `orderby` column
    filterby
        Column by which to filter the interactions
    filter_lambda
        If `filterby` is not `None`, provide a simple lambda function by which
        to filter the interactions to be plotted
    inverse_colour
        Whether to -log10 the `colour` column for plotting. `False` by default.
    inverse_size
        Whether to -log10 the `size` column for plotting. `False` by default.
    size_range
        Define size range - (min, max). Default is (2, 9)
    cmap
        Colour map to use for plotting. Default is 'viridis'
    figure_size
        Figure x,y size
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
                                ligand_complex = ligand_complex, 
                                receptor_complex = receptor_complex,
                                uns_key=uns_key
                                )
    _check_var(liana_res, var=colour, var_name='colour')
    _check_var(liana_res, var=size, var_name='size')

    liana_res = _filter_by(liana_res, filterby, filter_lambda)
    liana_res = _get_top_n(liana_res, top_n, orderby, orderby_ascending, orderby_absolute)
        
    # inverse sc if needed
    if inverse_colour:
        liana_res[colour] = _inverse_scores(liana_res[colour])
    if inverse_size:
        liana_res[size] = _inverse_scores(liana_res[size])

    # generate plot
    p = (ggplot(liana_res, aes(x='target', y='interaction', colour=colour, size=size))
         + geom_point()
         + facet_grid('~source')
         + scale_size_continuous(range=size_range)
         + scale_color_cmap(cmap)
         + labs(color=str.capitalize(colour),
                size=str.capitalize(size),
                y="Interactions (Ligand -> Receptor)",
                x="Target",
                title="Source")
         + theme_bw()
         + theme(legend_text=element_text(size=14),
                 strip_background=element_rect(fill="white"),
                 strip_text=element_text(size=15, colour="black"),
                 axis_text_y=element_text(size=10, colour="black"),
                 axis_title_y=element_text(colour="#808080", face="bold", size=15),
                 axis_text_x=element_text(size=11, face="bold", angle=90),
                 figure_size=figure_size,
                 plot_title=element_text(vjust=0, hjust=0.5, face="bold",
                                         colour="#808080", size=15)
                 )
         )

    if return_fig:
        return p

    p.draw()

    
def dotplot_by_sample(adata: anndata.AnnData  = None,
                      uns_key: str = 'liana_res',
                      liana_res: pandas.DataFrame = None,
                      sample_key: str = 'sample',
                      colour: str  = None,
                      size: str = None,
                      inverse_colour: bool = False,
                      inverse_size: bool = False,
                      source_labels: str | None = None,
                      target_labels: str | None = None,
                      ligand_complex: str | None = None, 
                      receptor_complex: str | None = None,
                      size_range: tuple = (2, 9),
                      cmap: str = 'viridis',
                      figure_size: tuple = (8, 6),
                      return_fig: bool = True
                      ):
    """
    A dotplot of interactions by sample
    
    Parameters
    ----------
        adata
            adata object with liana_res and  in adata.uns. Defaults to None.
        uns_key
            key in adata.uns where liana_res is stored. Defaults to 'liana_res'.
        liana_res
            liana_res a DataFrame in liana's format. Defaults to None.
        sample_key
            sample_key used to group different samples/contexts from `liana_res`. Defaults to 'sample'.
        colour
            `column` in `liana_res` to define the colours of the dots. Defaults to None.
        size
            `column` in `liana_res` to define the size of the dots. Defaults to None.
        inverse_colour
            Whether to -log10 the `colour` column for plotting. `False` by default. Defaults to False.
        inverse_size
            Whether to -log10 the `size` column for plotting. `False` by default. Defaults to False.
        source_labels
            `list` with keys as `source` and values as `label` to be used in the plot. Defaults to None.
        target_labels
            `list` with keys as `target` and values as `label` to be used in the plot. Defaults to None.
        ligand_complex
            `list` of ligand complexes to filter the interactions to be plotted. Defaults to None.
        receptor_complex
            `list` of receptor complexes to filter the interactions to be plotted. Defaults to None.
        size_range
            Define size range - (min, max). Default is (2, 9). Defaults to (2, 9).
        figure_size
            Figure x,y size. Defaults to (8, 6).
        return_fig
            `bool` whether to return the fig object, `False` only plots. Defaults to True.
        
    Returns
    -------
    Returns a dotplot of Class ggplot for the specified interactions by sample.
    
    """
    
    liana_res = _prep_liana_res(adata=adata,
                                liana_res=liana_res, 
                                source_labels=source_labels,
                                target_labels=target_labels,
                                ligand_complex=ligand_complex,
                                receptor_complex=receptor_complex,
                                uns_key=uns_key)
    _check_var(liana_res, var=colour, var_name='colour')
    _check_var(liana_res, var=size, var_name='size')
        
    # inverse sc if needed
    if inverse_colour:
        liana_res[colour] = _inverse_scores(liana_res[colour])
    if inverse_size:
        liana_res[size] = _inverse_scores(liana_res[size])

    p = (ggplot(liana_res, aes(x='target', y='source', colour=colour, size=size))
            + geom_point()
            + facet_grid(f'interaction~{sample_key}', space='free', scales='free')
            + scale_size_continuous(range=size_range)
            + scale_color_cmap(name=cmap)
            + labs(color=str.capitalize(colour),
                   size=str.capitalize(size),
                   y="Source",
                   x="Target",
                   title=sample_key.capitalize())
            + theme_bw()
            + theme(legend_text=element_text(size=14),
                    strip_background=element_rect(fill="white"),
                    strip_text=element_text(size=13, colour="black", angle=90),
                    axis_text_y=element_text(size=10, colour="black"),
                    axis_title_y=element_text(colour="#808080", face="bold", size=12),
                    axis_text_x=element_text(size=11, face="bold", angle=90),
                    axis_title_x=element_text(colour="#808080", face="bold", size=12),
                    figure_size=figure_size,
                    plot_title=element_text(vjust=0, hjust=0.5, face="bold", size=12),
                    )
            )
    if return_fig:
        return p
    
    p.draw()


