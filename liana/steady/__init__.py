from .liana_pipe import liana_pipe
from .get_mean_perms import get_means_perms
from .scores.cellphonedb import cellphonedb
from .scores.natmi import natmi
from .scores.singlecellsignalr import singlecellsignalr
from .scores.connectome import connectome
from .scores.logfc import logfc
from .Method import MethodMeta
from .scores.consensus import ConsensusClass, _consensus_meta

# callable consensus instance
_methods = [cellphonedb, connectome, logfc, natmi, singlecellsignalr]
consensus = ConsensusClass(_consensus_meta, methods=_methods)
