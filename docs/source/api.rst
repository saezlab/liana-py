API
====================================================
   
Import liana as::
   import liana as li

Methods:
------------------------------------------------------------

Objects of class Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: liana
.. currentmodule:: liana

.. autosummary::
   :toctree: api

   method.cellphonedb
   method.connectome
   method.logfc
   method.natmi
   method.singlecellsignalr
   method.rank_aggregate


Callable Method instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Method implementations are callable via their following instances:

.. module:: liana
.. currentmodule:: liana

.. autosummary::
   :toctree: generated

   method.cellphonedb.__call__
   method.connectome.__call__
   method.logfc.__call__
   method.natmi.__call__
   method.singlecellsignalr.__call__


Running Rank Aggregate Instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated

   method.rank_aggregate.__call__


LIANA's Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`liana` relies on a common pipeline to generate
the statistics for all of the methods above. This enables
the straightforward addition to any novel method.

.. autosummary::
   :toctree: generated

   method.liana_pipe


Visualization options:
------------------------------------------------------------
.. module:: liana.plotting
.. currentmodule:: liana

.. autosummary::
   :toctree: generated

   plotting.dotplot


General utils:
---------------------------------------------------------------

.. autosummary::
   :toctree: generated

   liana.resource.select_resource
   liana.resource.show_resources
   liana.method.show_methods
