import importlib.metadata
__version__ = importlib.metadata.version("liana")

"""actual API"""
from . import _logging, testing, multi
from . import method as mt
from . import plotting as pl
from . import resource as rs
from . import utils as ut

# done after everything has been imported (adapted from scanpy)
import sys
from scanpy._utils import annotate_doc_types

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['mt', 'pl', 'rs', 'ut']})
annotate_doc_types(sys.modules[__name__], 'liana')

del sys, annotate_doc_types
