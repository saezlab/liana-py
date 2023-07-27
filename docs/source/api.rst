.. LiAna API Documentation
   ===============================

Import liana as::

   import liana as li

Single-cell
----------------------------------

Ligand-Receptor Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods are implemented as instances of the same Method class.
Each instance provides helper functions and consistent attributes,
to describe each method instance.

.. autosummary::
   :toctree: api

   liana.method.cellchat
   liana.method.cellphonedb
   liana.method.connectome
   liana.method.logfc
   liana.method.natmi
   liana.method.singlecellsignalr
   liana.method.geometric_mean
   liana.method.rank_aggregate

Callable Method instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With each Method (instance of class Method) being callable:

.. autosummary::
   :toctree: api

   liana.method.cellchat.__call__
   liana.method.cellphonedb.__call__
   liana.method.connectome.__call__
   liana.method.logfc.__call__
   liana.method.natmi.__call__
   liana.method.singlecellsignalr.__call__
   liana.method.geometric_mean.__call__
   liana.method.rank_aggregate.__call__

Spatial
----------------------------------

Local bivariate metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api
   
   liana.method.bivar
   liana.method.lr_bivar

Learn Spatial Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api
   
   liana.method.MistyData
   liana.method.genericMistyData
   liana.method.lrMistyData

Multi-Sample
----------------------------------

.. autosummary::
   :toctree: api

   liana.plotting.df_to_lr
   liana.plotting.to_tensor_c2c
   liana.plotting.adata_to_views
   liana.plotting.lrs_to_views
   liana.plotting.nmf
   liana.plotting.estimate_elbow

Visualization options
----------------------------------

.. autosummary::
   :toctree: api

   liana.plotting.dotplot
   liana.plotting.dotplot_by_sample
   liana.plotting.tileplot
   liana.plotting.connectivity
   liana.plotting.target_metrics
   liana.plotting.contributions
   liana.plotting.interactions

Utility functions
----------------------------------

.. autosummary::
   :toctree: api

   liana.utils.obsm_to_adata
   liana.utils.mdata_to_anndata
   liana.utils.zi_minmax
   liana.utils.neg_to_zero
   liana.utils.spatial_neighbors
   liana.utils.get_factor_scores
   liana.utils.get_variable_loadings

Prior knowledge
----------------------------------

.. autosummary::
   :toctree: api

   liana.resource.select_resource
   liana.resource.show_resources
   liana.resource.generate_lr_geneset

Intracellular
----------------------------------

.. autosummary::
   :toctree: api

   liana.method.causalnet
