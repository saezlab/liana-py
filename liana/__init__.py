__version__ = '0.0.1'  # noqa: F401
__version_info__ = tuple([int(num) for num in __version__.split('.')])  # noqa: F401


import liana.plotting as pl
import steady as sc
from .steady import liana_pipe, cellphonedb, natmi, singlecellsignalr,\
    connectome, logfc, rank_aggregate, show_methods
from .resource.select_resource import select_resource, show_resources