import numpy as np

from plotnine import ggplot, geom_point, aes, \
    facet_grid, labs, theme_bw, theme, element_text, element_rect


def dotplot(adata, colour, size, source_labels=None,
            target_labels=None, top_n=None,
            orderby=None, orderby_ascending=False,
            filterby=None, filter_lambda=None,
            inverse_colour=False, inverse_size=False, figure_size=(8, 6),
            return_fig=True) -> ggplot:
    """
    Dotplot interactions by source and target cells

    Parameters
    ----------
    adata
        `AnnData` object with `liana_res` in `adata.uns`
    colour

    size
    source_labels
    target_labels
    top_n
    orderby
    orderby_ascending
    filterby
    filter_lambda
    inverse_colour
    inverse_size
    figure_size
    return_fig

    Returns
    -------

    """
    assert 'liana_res' in adata.uns_keys()

    # extract results & create interaction col
    liana_res = adata.uns['liana_res'].copy()
    liana_res['interaction'] = liana_res.ligand_complex + ' -> ' + liana_res.receptor_complex

    # subset to only cell labels of interest
    if source_labels is not None:
        source_msk = np.isin(liana_res.source, source_labels)
        liana_res = liana_res[source_msk]
    if target_labels is not None:
        target_msk = np.isin(liana_res.target, target_labels)
        liana_res = liana_res[target_msk]

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
         + labs(color=str.capitalize(colour),
                size=str.capitalize(size),
                y="Interactions (Ligand -> Receptor)",
                x="Target",
                title="Source")
         + theme_bw()
         + theme(legend_text=element_text(size=14),
                 strip_background=element_rect(fill="white"),
                 strip_text=element_text(size=15, colour="black"),
                 axis_text_x=element_text(size=11, face="bold"),
                 figure_size=figure_size,
                 plot_title=element_text(vjust=0, hjust=0.5, face="bold",
                                         colour="#808080", size=15),
                 axis_title=element_text(colour="#808080", face="bold", size=15),
                 axis_text_y=element_text(size=10, colour="black")
                 )
         )

    if return_fig:
        return p

    p.draw()


def _aggregate_scores(res, what, how, entities):
    return res.groupby(entities).agg(score=(what, how)).reset_index()


def _inverse_scores(score):
    return -np.log10(score + np.finfo(float).eps)
