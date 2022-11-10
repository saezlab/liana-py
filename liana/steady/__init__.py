from .liana_pipe import liana_pipe
from .get_mean_perms import get_means_perms
from .sc.cellphonedb import cellphonedb
from .sc.natmi import natmi
from .sc.singlecellsignalr import singlecellsignalr
from .sc.connectome import connectome
from .sc.logfc import logfc
from .Method import MethodMeta, _show_methods
from .sc.rank_aggregate import ConsensusClass, _consensus_meta

# callable consensus instance
_methods = [cellphonedb, connectome, logfc, natmi, singlecellsignalr]
rank_aggregate = ConsensusClass(_consensus_meta, methods=_methods)


def show_methods():
    """Shows methods available in LIANA"""
    return _show_methods(_methods + [rank_aggregate])
