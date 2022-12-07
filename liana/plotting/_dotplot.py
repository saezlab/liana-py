import anndata
import numpy as np
import pandas

from plotnine import ggplot, geom_point, aes, \
    facet_grid, labs, theme_bw, theme, element_text, element_rect, scale_size_continuous


def dotplot(adata: anndata.AnnData = None,
            liana_res: pandas.DataFrame = None,
            colour: str = None, size: str = None,
            source_labels: list = None, target_labels: list = None,
            top_n: int = None, orderby: (str, None) = None,
            orderby_ascending: (bool, None) = None,
            filterby: (bool, None) = None, filter_lambda=None,
            inverse_colour: bool = False, inverse_size: bool = False,
            size_range: tuple = (2, 9),
            figure_size: tuple = (8, 6),
            return_fig=True) -> ggplot:
    """
    Dotplot interactions by source and target cells

    Parameters
    ----------
    adata
        `AnnData` object with `liana_res` in `adata.uns`. Default is `None`.
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
    figure_size
        Figure x,y size
    return_fig
        `bool` whether to return the fig object, `False` only plots

    Returns
    -------
    A `plotnine.ggplot` instance

    """
    if (liana_res is not None) & (adata is not None):
        raise AttributeError('Ambiguous! One of `liana_res` or `adata` should be provided.')
    if adata is not None:
        assert 'liana_res' in adata.uns_keys()
        liana_res = adata.uns['liana_res'].copy()
    if liana_res is not None:
        liana_res = liana_res.copy()
    if (liana_res is None) & (adata is None):
        raise ValueError('`liana_res` or `adata` must be provided!')
    if (colour is None) | (size is None):
        raise ValueError('`colour` and `size` must be provided!')
    if colour not in liana_res.columns:
        raise ValueError('`colour` column was not found in `liana_res`!')
    if size not in liana_res.columns:
        raise ValueError('`size` column was not found in `liana_res`!')

    liana_res['interaction'] = liana_res.ligand_complex + ' -> ' + liana_res.receptor_complex

    # subset to only cell labels of interest
    if source_labels is not None:
        source_msk = np.isin(liana_res.source, source_labels)
        liana_res = liana_res[source_msk]
        possible_sources = np.unique(liana_res['source'])
        covered = np.isin(source_labels, possible_sources)
        if not all(covered):
            not_covered = np.array(source_labels)[~covered]
            raise ValueError(f"{not_covered} not found in `liana_res['source']`!")
    if target_labels is not None:
        target_msk = np.isin(liana_res.target, target_labels)
        liana_res = liana_res[target_msk]
        possible_targets = np.unique(liana_res['target'])
        covered = np.isin(target_labels, possible_targets)
        if not all(covered):
            not_covered = np.array(target_labels)[~covered]
            raise ValueError(f"{not_covered} not found in `liana_res['target']`!")

    if filterby is not None:
        msk = liana_res[filterby].apply(filter_lambda)
        relevant_interactions = np.unique(liana_res[msk].interaction)
        liana_res = liana_res[np.isin(liana_res.interaction, relevant_interactions)]

    # inverse sc if needed
    if inverse_colour:
        liana_res[colour] = _inverse_scores(liana_res[colour])
    if inverse_size:
        liana_res[size] = _inverse_scores(liana_res[size])

    if top_n is not None:
        # get the top_n for each interaction
        if orderby is None:
            ValueError("Please specify the column to order the interactions.")
        if orderby_ascending is None:
            ValueError("Please specify if `orderby` is ascending or not.")
        if orderby_ascending:
            how = 'min'
        else:
            how = 'max'
        top_lrs = _aggregate_scores(liana_res, what=orderby, how=how,
                                    entities=['interaction',
                                              'ligand_complex',
                                              'receptor_complex']
                                    )
        top_lrs = top_lrs.sort_values('score', ascending=orderby_ascending).head(top_n).interaction
        # Filter liana_res to the interactions in top_lrs
        liana_res = liana_res[liana_res.interaction.isin(top_lrs)]

    # generate plot
    p = (ggplot(liana_res, aes(x='target', y='interaction', colour=colour, size=size))
         + geom_point()
         + facet_grid('~source')
         + scale_size_continuous(range=size_range)
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


def _aggregate_scores(res, what, how, entities):
    return res.groupby(entities).agg(score=(what, how)).reset_index()


def _inverse_scores(score):
    return -np.log10(score + np.finfo(float).eps)
