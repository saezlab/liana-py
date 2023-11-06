import pandas as pd
import plotnine as p9

from liana._constants import Keys as K, DefaultValues as V
from liana._docs import d

@d.dedent
def target_metrics(misty,
                   stat,
                   top_n = None,
                   ascending = False,
                   key = None,
                   filterby = None,
                   filter_lambda: callable = None,
                   figure_size: tuple = (7,5),
                   return_fig: bool = V.return_fig):
    """
    Plot target metrics.

    Parameters
    ----------
    %(misty)s
    stat : str
        Statistic to plot
    %(top_n)s
    ascending : bool
        Whether to sort in ascending order
    key : callable
        Function to use to sort the dataframe
    %(filterby)s
    %(filter_lambda)s
    %(figure_size)s
    %(return_fig)s

    Returns
    -------

    Returns a plotnine plot.

    """
    target_metrics = misty.uns[K.target_metrics].copy()

    if filterby is not None:
        msk = target_metrics[filterby].apply(filter_lambda)
        target_metrics = target_metrics[msk]
    if top_n is not None:
        target_metrics = target_metrics.sort_values(stat, ascending=ascending, key=key).head(top_n)

    # get order of target by decreasing intra.R2
    targets = target_metrics.sort_values(by=stat, ascending=False)['target'].values
    # targets as categorical variable
    target_metrics['target'] = pd.Categorical(target_metrics['target'],
                                              categories=targets,
                                              ordered=True)

    p = (p9.ggplot(target_metrics, p9.aes(x='target', y=stat)) +
         p9.geom_point(size=3) +
         p9.theme_bw() +
         p9.theme(axis_text_x=p9.element_text(rotation=90),
                  figure_size=figure_size) +
        p9.labs(x='Target')
        )

    if return_fig:
        return p
    p.draw()

@d.dedent
def contributions(misty,
                  top_n=None,
                  ascending=False,
                  key=None,
                  figure_size: tuple = (7, 5),
                  return_fig: bool = V.return_fig):
    """
    Plot view contributions per target.

    Parameters
    ----------

    %(misty)s
    %(top_n)s
    ascending : bool
        Whether to sort in ascending order
    key : callable
        Function to use to sort the dataframe
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    A plotnine plot.

    """
    target_metrics = misty.uns[K.target_metrics].copy()

    view_names = misty.view_names.copy()
    if 'intra' not in target_metrics.columns:
        view_names.remove('intra')

    target_metrics = target_metrics[['target', *view_names]]
    target_metrics = target_metrics.melt(id_vars='target', var_name='view', value_name='contribution')

    if top_n is not None:
        target_metrics = target_metrics.sort_values('contribution', ascending=ascending, key=key).head(top_n)

    p = (p9.ggplot(target_metrics, p9.aes(x='target', y='contribution', fill='view')) +
            p9.geom_bar(stat='identity') +
            p9.theme_bw(base_size=14) +
            p9.theme(axis_text_x=p9.element_text(rotation=90),
                     figure_size=figure_size) +
            p9.scale_fill_brewer(palette=2, type='qual') +
            p9.labs(x='Target', y='Contribution', fill='View')
    )

    if return_fig:
        return p
    p.draw()

@d.dedent
def interactions(misty,
                 view,
                 top_n = None,
                 ascending = False,
                 key = None,
                 filterby = None,
                 filter_lambda: callable = None,
                 figure_size: tuple = (7,5),
                 return_fig: bool = V.return_fig):
    """
    Plot interaction importances.

    Parameters
    ----------

    %(misty)s
    view : str
        A view to plot
    %(top_n)s
    ascending : bool
        Whether to sort interactions in ascending order
    key : str
        Key to use when sorting interactions
    %(filterby)s
    %(filter_lambda)s
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    A plotnine plot.

    """
    interactions = misty.uns[K.interactions].copy()
    interactions = interactions[interactions['view'] == view]
    grouped = interactions.groupby('predictor')['importances'].apply(lambda x: x.isna().all())
    interactions = interactions[~interactions['predictor'].isin(grouped[grouped].index)]

    if filterby is not None:
        top_interactions = interactions[interactions[filterby].apply(filter_lambda)]
        top_interactions = top_interactions.drop_duplicates(['target', 'predictor'])
    if top_n is not None:
        interactions = interactions.sort_values(by='importances', key=key, ascending=ascending)
        top_interactions = interactions.drop_duplicates(['target', 'predictor']).head(top_n)

    if (filterby is not None) or (top_n is not None):
        interactions = interactions[interactions['target'].isin(top_interactions['target']) &
                            interactions['predictor'].isin(top_interactions['predictor'])]

    p = (p9.ggplot(interactions,
                   p9.aes(x='predictor',
                          y='target',
                          fill='importances')
                   ) +
    p9.geom_tile() +
    p9.theme_minimal(base_size=12) +
    p9.theme(axis_text_x=p9.element_text(rotation=90),
             figure_size=figure_size) +
    p9.labs(x='Predictor', y='Target', fill='Importance')
    )

    if return_fig:
        return p
    p.draw()
