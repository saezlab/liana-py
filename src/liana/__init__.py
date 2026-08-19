__version__ = '1.8.1'

# done after everything has been imported (adapted from scanpy)
import sys

from scanpy._utils import annotate_doc_types

from liana import method as mt
from liana import multi as mu
from liana import plotting as pl
from liana import resource as rs
from liana import testing
from liana import utils as ut

__all__ = ['mt', 'mu', 'pl', 'rs', 'testing', 'ut']

# register the short aliases as modules too, so `import liana.mt` works and not
# only `liana.mt` after `import liana`. `mu` was missing, so `import liana.mu`
# raised ModuleNotFoundError while its four siblings resolved.
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in __all__})
annotate_doc_types(sys.modules[__name__], 'liana')

del sys, annotate_doc_types
