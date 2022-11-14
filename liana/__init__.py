__version__ = '0.0.2'
__version_info__ = tuple([int(num) for num in __version__.split('.')])

from .method import liana_pipe, cellphonedb, natmi, singlecellsignalr, \
    connectome, logfc, rank_aggregate, show_methods
from .resource.select_resource import select_resource, show_resources
import liana.plotting as pl
