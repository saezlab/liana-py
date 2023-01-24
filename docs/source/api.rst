API
====================================================
   
Import liana as::
   import liana as li

Methods:
------------------------------------------------------------


Instances of Method Class
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



Visualization options:
------------------------------------------------------------
.. module:: liana.plotting
.. currentmodule:: liana

.. autosummary::
   :toctree: api

   plotting.dotplot
   plotting.dotplot_by_sample


General utils:
---------------------------------------------------------------

.. autosummary::
   :toctree: api

   resource.select_resource
   resource.show_resources
   method.show_methods


Multi-sample utils:
---------------------------------------------------------------
liana also provides utilities to work with multiple samples,
such as running any method by sample using the `by_sample` function:

.. module:: liana
.. currentmodule:: liana


.. autosummary::
   :toctree: api

   method.Method.by_sample


and converting the output of this function to Tensor-cell2cell format.

.. autosummary::
   :toctree: api

   multi.to_tensor_c2c



Functional utils:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LIANA also utility function to convert e.g. geneset resources
to a format that can be used to do enrichment analysis at 
the ligand-receptor space.

.. module:: liana
.. currentmodule:: liana


.. autosummary::
   :toctree: api

   funcomics.generate_lr_geneset


Spatial utils:
---------------------------------------------------------------

.. autosummary::
   :toctree: api

   liana.method.get_spatial_proximity



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
