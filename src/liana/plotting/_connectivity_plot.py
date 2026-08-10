import pandas as pd
from anndata import AnnData
from matplotlib.figure import Figure
from plotnine import aes, geom_point, ggplot, labs, theme, theme_minimal

from liana._constants import DefaultValues as V
from liana._constants import Keys as K
from liana._docs import d
from liana._logging import _logg


@d.dedent
def connectivity(adata: AnnData,
                 idx: int,
                 spatial_key: str = K.spatial_key,
                 connectivity_key: str = K.connectivity_key,
                 size: float = 1,
                 figure_size: tuple[float, float] = (5.4, 5),
                 return_fig: bool = V.return_fig
                 ) -> Figure:
    """
    Plot spatial connectivity weights.

    Parameters
    ----------
    %(adata)s
    idx
        Column index of the connectivity weights to plot.
    %(spatial_key)s
    %(connectivity_key)s
    size
        Size of the points
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    The resulting connectivity plot.

    Raises
    ------
    AssertionError
        If `connectivity_key` or `spatial_key` are not in `adata.obsp` or `adata.obsm` respectively.

    """
    assert connectivity_key in list(adata.obsp.keys())
    assert spatial_key in adata.obsm_keys()

    _logg("This function will be deprecated in the next version. " +
          "Please use scanpy or squidpy for plotting spatial connectivities.", level='warn')

    coordinates = pd.DataFrame(adata.obsm[spatial_key],
                               index=adata.obs_names,
                               columns=['x', 'y']).copy()
    coordinates['connectivity'] = adata.obsp[connectivity_key][:, idx].toarray()
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
