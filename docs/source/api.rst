API
===============================

Import liana as::

   import liana as li


Single-cell
----------------------------------

.. module:: liana
.. currentmodule:: liana


Callable Ligand-Receptor Method instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ligand-receptor method instances provide helper functions and consistent attributes,
to describe each method instance, and are callable:

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

   liana.method.bivariate.__call__


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

   liana.multi.df_to_lr
   liana.multi.to_tensor_c2c
   liana.multi.adata_to_views
   liana.multi.lrs_to_views
   liana.multi.nmf
   liana.multi.estimate_elbow


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
   liana.utils.interpolate_adata

Prior knowledge
----------------------------------

.. autosummary::
   :toctree: api

   liana.resource.select_resource
   liana.resource.show_resources
   liana.resource.generate_lr_geneset
   liana.resource.explode_complexes
   liana.resource.get_metalinks
   liana.resource.describe_metalinks
   liana.resource.get_metalinks_values

Intracellular
----------------------------------

.. autosummary::
   :toctree: api

   liana.method.find_causalnet
   liana.method.build_prior_network
   liana.method.estimate_metalinks
