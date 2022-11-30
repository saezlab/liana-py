from plotnine import ggplot, aes, geom_point, theme_minimal, labs
import anndata
import pandas as pd


def proximity_plot(adata: anndata.AnnData, idx: int, return_fig: bool = True):
    """
    Plot spatial proximity weights.

    Parameters
    ----------
    adata
        `AnnData` object with `proximity` (spatial proximity weights) in `adata.obsm`.
    idx
        Spot/cell index
    return_fig
        `bool` whether to return the fig object, `False` only plots

    Returns
    -------
    A `plotnine.ggplot` instance

    """

    assert 'proximity' in adata.obsm_keys()
    assert 'spatial' in adata.obsm_keys()

    coordinates = pd.DataFrame(adata.obsm['spatial'],
                               index=adata.obs_names,
                               columns=['x', 'y']).copy()
    coordinates['proximity'] = adata.obsm['proximity'][:, idx].A

    p = (ggplot(coordinates.sort_values('proximity', ascending=True),
                aes(x='x', y='y', colour='proximity'))
         + geom_point(size=2.7, shape='8')
         + theme_minimal()
         + labs(colour='Proximity', y='y Coordinate', x='x Coordinate')
         )

    if return_fig:
        return p

    p.draw()
