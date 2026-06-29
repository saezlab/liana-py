__version__ = '1.8.0'

# done after everything has been imported (adapted from scanpy)
import sys

from scanpy._utils import annotate_doc_types

from liana import method as mt
from liana import multi as mu
from liana import plotting as pl
from liana import resource as rs
from liana import testing
from liana import utils as ut

sys.modules.update({f'{__name__}.{m}': globals()[m]
                    for m in ['mt', 'pl', 'rs', 'ut']})
annotate_doc_types(sys.modules[__name__], 'liana')

del sys, annotate_doc_types
