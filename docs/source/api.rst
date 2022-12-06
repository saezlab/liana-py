API
====================================================
   
Import liana as::
   import liana as li

Methods:
------------------------------------------------------------


Callable Method instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With each Method (instance of class Method) being callable:

.. module:: liana
.. currentmodule:: liana

.. autosummary::
   :toctree: api

   method.cellchat.__call__
   method.cellphonedb.__call__
   method.connectome.__call__
   method.logfc.__call__
   method.natmi.__call__
   method.singlecellsignalr.__call__
   method.geometric_mean.__call__


Callable Rank Aggregate Instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LIANA's RankAggregate Class is a child of the Method class,
and hence shares the same attributes and functions.

The RankAggregate Class can be used to generate a consensus
for any of the methods in LIANA. Similarly to any other
Method instance, it is also callable:

.. autosummary::
   :toctree: api

   method.rank_aggregate.__call__


LIANA's Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`liana` relies on a common pipeline to generate
the statistics for all of the methods above. This enables
the straightforward inclusion to any additional method to `liana`.

.. autosummary::
   :toctree: api

   method.liana_pipe


Visualization options:
------------------------------------------------------------
.. module:: liana.plotting
.. currentmodule:: liana

.. autosummary::
   :toctree: api

   plotting.dotplot


General utils:
---------------------------------------------------------------

.. autosummary::
   :toctree: api

   liana.resource.select_resource
   liana.resource.show_resources
   liana.method.show_methods


All instances of Method Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Methods are implemented as instances of the same Method class.
Each instance provides helper functions and consistent attributes,
to describe each method instance.

.. module:: liana
.. currentmodule:: liana

.. autosummary::
   :toctree: api

   method.cellchat
   method.cellphonedb
   method.connectome
   method.logfc
   method.natmi
   method.singlecellsignalr
   method.geometric_mean
   method.rank_aggregate
