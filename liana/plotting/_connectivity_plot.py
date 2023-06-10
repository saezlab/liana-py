from plotnine import ggplot, aes, geom_point, theme_minimal, labs, theme
import anndata
import pandas as pd


def connectivity(adata: anndata.AnnData, idx: int, spatial_key = 'spatial', connectivity_key = 'spatial_connectivities', size=1.5, figure_size=(5.4, 5), return_fig: bool = True):
    """
    Plot spatial connectivity weights.

    Parameters
    ----------
    adata
        `AnnData` object with `connectivity` (spatial connectivity weights) in `adata.obsm`.
    idx
        Spot/cell index
    spatial_key
        Key to use to retrieve the spatial coordinates from adata.obsm.
    connectivity_key
        Key to use to retrieve the connectivity (sparse) matrix from adata.obsp.
    size
        Size of the points
    figure_size
        Size of the figure
    return_fig
        `bool` whether to return the fig object, `False` only plots

    Returns
    -------
    A `plotnine.ggplot` instance

    """

    assert connectivity_key in list(adata.obsp.keys())
    assert spatial_key in adata.obsm_keys()

    coordinates = pd.DataFrame(adata.obsm[spatial_key],
                               index=adata.obs_names,
                               columns=['x', 'y']).copy()
    coordinates['connectivity'] = adata.obsp[connectivity_key][:, idx].A
    coordinates['y'] = coordinates['y'].max() - coordinates['y'] # flip y

    p = (ggplot(coordinates.sort_values('connectivity', ascending=True),
                aes(x='x', y='y', colour='connectivity'))
         + geom_point(size=size, shape='8')
         + theme_minimal()
         + labs(colour='connectivity', y='y Coordinate', x='x Coordinate')
         + theme(figure_size=figure_size)
         ) 


    if return_fig:
        return p

    p.draw()
