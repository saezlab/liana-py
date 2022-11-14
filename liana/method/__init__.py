from ._liana_pipe import liana_pipe
from ._Method import MethodMeta, _show_methods
from .sc._rank_aggregate import AggregateClass, _rank_aggregate_meta
from .sc import cellphonedb, connectome, logfc, natmi, singlecellsignalr

# callable consensus instance
_methods = [cellphonedb, connectome, logfc, natmi, singlecellsignalr]
rank_aggregate = AggregateClass(_rank_aggregate_meta, methods=_methods)


def show_methods():
    """Shows methods available in LIANA"""
    return _show_methods(_methods + [rank_aggregate])
