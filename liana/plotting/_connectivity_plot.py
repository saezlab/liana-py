from plotnine import ggplot, aes, geom_point, theme_minimal, labs, theme
import anndata
import pandas as pd

from liana._logging import _logg
from liana._docs import d
from liana._constants import Keys as K, DefaultValues as V

@d.dedent
def connectivity(adata: anndata.AnnData,
                 idx: int,
                 spatial_key=K.spatial_key,
                 connectivity_key=K.connectivity_key,
                 size=1,
                 figure_size=(5.4, 5),
                 return_fig: bool = V.return_fig):
    """
    Plot spatial connectivity weights.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(connectivity_key)s
    size
        Size of the points
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    A `plotnine.ggplot` instance

    """

    assert connectivity_key in list(adata.obsp.keys())
    assert spatial_key in adata.obsm_keys()

    _logg("This function will be deprecated in the next version. " +
          "Please use scanpy or squidpy for plotting spatial connectivities.", level='warn')

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
