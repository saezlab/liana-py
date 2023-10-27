Release notes
=============

1.0.3 (XX.11.2023)
-------------------------------------------------
- Added `filterby` and `filter_lambda` parameters to `li.pl.interactions` and `li.pl.target_metrics` to allow filtering of interactions and metrics, respectively.

- Removed unnecessary `stat` parameter from `li.pl.contributions`

- Added tests to ensure both `lr_bivar` and single-cell methods throw an exception when the resource is not covered by the data.

- `estimate_elbow` will add the errors and the number of patterns to `.uns`

- When `groupby` or `sample_key` are not categorical liana will now print a warning before converting them to categorical. Related to #28

- Various Documentation improvements

- Renamed some `*_separator` columns to `*_sep` for consistency


1.0.2 (13.10.2023)
-------------------------------------------------
- Added as `seed` param to `find_causalnet`, used to a small amount of noise to the nodes in to avoid obtaining multiple solutions to the same problem when multiple equal solutions are possible.

- Updated `installation.rst` to refer to `pip install liana[common]` and `liana[full]` for extended installations.

- Fixed a bug which would cause `bivar` to crash when an AnnData object was passed

Merged #61 including the following:

- Added `standardize` parameter to spatial_neighbors, used to standardize the spatial connectivities such that each spot's proximity weights to 1. Required for non-standardized metrics (such as `product`)

- Fixed edge case in `assert_covered` to handle interactions not present in `adata` nor the resource.

- Added simple product (scores ranging from -inf, +inf) and norm_product (scores ranging from -1, +1). 
The former is a simple product of x and y, while the latter standardized each variable to be between 0 and 1, following weighing by spatial proximity, and then multiplies them.
Essentially, it diminishes the effect of spatial proximity on the score, while still taking it into account. We observed that this is useful for e.g. border zones.


1.0.1 Stable Release (30.09.2023)
-------------------------------------------------
- Bumped CORNETO version and it's now installed via PyPI.

1.0.0a2 (19.09.2023)

- Interactions names in `tileplot` and `dotplot` will now be sorted according to `orderby` when used; related to #55

- Added `filter_view_markers` function to filter view markers considered background in MOFAcellular tutorial

- Added `keep_stats` parameter to `adata_to_views` to enable pseudobulk stats to be kept.

- Replace `intra_groupby` and `extra_groupby` with `maskby` in misty. 
  The spots will now only be filtered according to `maskby`, such that both intra and extra both contain the same spots.
  The extra views are multiplied by the spatial connectivities prior to masking and the model being fit

- Merge MOFAcell improvements; related to #42 and #29

- Targets with zero variance will no longer be modeled by misty.

- Resolve #46 - refactored misty's pipeline

- Resolved logging and package import verbosity issues related to #43

- Iternal .obs['label'] placeholder renamed to the less generic .obs['@label']; related to #53

- Minor Readme & tutorial text improvements.


1.0.0a1 Biorxiv (30.07.2023)
---------------------------------------------------------

- `positive_only` in bivariate metrics was renamed to `mask_negatives` will now mask only negative-negative/low-low interactions, and not negative-positive interactions.

- Replaced MSigDB with transcription factor activities in MISTy's tutorial

- Enable sorting according to ascending order in misty-related plots

- Enable `cmap` to be passed to tileplot & dotplots

- Minor Readme & tutorial improvements.


1.0.0a0 (27.07.2023) LIANA+ Release
---------------------------------------------------------

LIANA becomes LIANA+.

Major changes have been made to the repository, however the API visible to the user should be largely consistent with previous versions, except minor exceptions:
- `li.fun.generate_lr_geneset` is now called via `li.rs.generate_lr_geneset`

- the old 'li.funcomics' model is now renamed to something more general: `li.utils`

- `get_factor_scores` and `get_variable_loadings` were moved to `li.utils`


LIANA+ includes the following new features:

Spatial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A sklearn-based implementation to learn spatially-informed multi-view models, i.e. [MISTy](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02663-5) models.

- A new tutorial that shows how to use LIANA+ to build and run MISTy models.

- Five vectorized local spatially-informed bivariate clustering and similarity metrics, such as [Moran's R](https://www.biorxiv.org/content/10.1101/2022.08.19.504616v1.full), Cosine, Jaccard, Pearson, Spearman. As well as a numba-compiled [Masked Spearman](https://www.nature.com/articles/s41592-020-0885-x) local score.

- A new tutorial that shows how to use LIANA+ to compute spatially-informed bivariate metrics, permutations-based p-values, interaction categoriez, as well as 
how to summarize those into patterns using NMF.

- A radial basis kernel is implemented to calculate spot/cell connectivities (spatial connectivities); this is used by the spatially-informed bivariate metrics and MISTy.
It mirrors [squidpy's](https://squidpy.readthedocs.io/en/stable/) `sq.gr.spatial_neighbors` function, and is hence interchangeable with it. 


Handling multiple modalities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- LIANA+ will now work with multi-modal data, i.e. it additionally support MuData objects as well as AnnData objects.
The API visible to the user is the same, but the underlying implementation is different.

- These come with a new tutorial that shows how to use LIANA+ with multi-modal (CITE-Seq) data, along with inbuilt transformations.

- The same API is also adapted by the local bivariate metrics, i.e. they can also be used with multi-modal data.


Multi-conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A utility function has been added that will take any dataframe with various statistics and append it to information from AnnData objects;
thus creating a multi-condition dataframe in the format of LIANA.

- A new tutorial that shows how to use PyDESeq2 together with this utility function has been added, essentially a tutorial on "Hypothesis-driven CCC".

Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A tileplot (`li.pl.tileplot`) has been added to better visualize ligands and receptors independently.

- MISTy-related visualizations have been added to vislualize view contributions and performance, and interaction coefficients/importances.

- A simple plot `li.pl.connectivity` is added to show spatial connectivities 

Others
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A Causal Network inference function has been added to infer downstream signalling networks. This is currently placed in the tutorial with PyDESeq2.

- An elbow approximation approach has been added to the NMF module, to help with the selection of the number of patterns.

- Various utility functions to simplify AnnData extraction/conversion, Matrix transformations, etc (added to `li.ut`)

Note: this is just an overview of the new features, for details please refer to the tutorials, API, and documentation.



0.1.9 (06.06.2023)
-----------------------------------------------------------------

- Fixed issues with deprecated params of pandas.DataFrame.to_csv & .assert_frame_equal in tests

- `multi.get_variable_loadings` will now return all factors

- Added source & target params to `fun.generate_lr_geneset`

- Refactored `sc._Method._get_means_perms` & related scoring functions to be more efficient.
 `None` can now be passed to n_perms to avoid permutations - these are only relevant if specificity is assumed to be relevant.

- LIANA's aggregate method can now be customized to include any method of choice (added an example to basic_usage).

- Removed 'Steady' aggregation from rank_aggregate

- Changed deprecated np.float to np.float32 in `liana_pipe`, relevant for CellChat `mat_max`.

- Method results will now be ordered by magnitude, if available, if not specificity is used.

- Added `ligand_complex` and `receptor_complex` filtering to liana's dotplot

- MOFAcellular will now work only with decoupler>=1.4.0 which implements edgeR-like filtering for the views.


0.1.8 (24.03.2023)
------------------------------------------------------------------------------------------------------------------------------

- Removed walrus operator to support Python 3.7

- Added a tutorial that shows the repurposed use of MOFA with liana to obtain intercellular communication programmes, inspired by Tensor-cell2cell

- Added a tutorial that shows the repurposed use of MOFA to the analysis of multicellular programmes as in Ramirez et al., 2023

- Added `key_added` parameter to save liana results to any `adata.uns`` slot, and `uns_key` to use liana results from any `adata.uns` slot

- `inplace` now works as intended (i.e. only writes to `adata.uns` if `inplace` is True).


0.1.7 (08.02.2023)
------------------------------------------------------------------------------------------------------------------------------

- Fixed an edge case where subunits within the same complex with identical values resulted in duplicates. These are now arbitrarily removed according to random order.

- All methods' complexes will now be re-assembled according to the closest stat to expression that each method uses, e.g. `cellchat` will use `trimeans` and the rest `means`.

- Added a basic liana to Tensor-cell2cell tutorial as a solution to liana issue #5

- Updated the basic tutorial 

- Referred to CCC chapter from Theis' best-practices book


0.1.6 (23.01.2023)
-----------------------------------------
- Fixed issue with duplicate subunits for non-expressed LRs when `return_all_lrs` is True

- `min_prop` when working with `return_all_lrs` is now filled with 0s

- Added `by_sample` function to class Method that returns a long-format dataframe of ligand-receptors, for each sample

- Added `dotplot_by_sample` function to visualize ligand-receptor interactions across samples

- Refractored preprocessing of `dotplot` and `dotplot_by_sample` to a separate function

- Changed "pvals" of geometric_mean method to "gmean_pvals" for consistency

- `to_tensor_c2c` utility function to convert a long-format dataframe of ligand-receptor interactions by sample to Tensor-cell2cell tensor.

- Added a list to track the instances of `MethodMeta` class

- Added `generate_lr_geneset` function to generate a geneset of ligand-receptors for different prior knowledge databases


0.1.5 (11.01.2023)
-----------------------------------------
- Hotfix `return_all_lrs` specificity_rank being assigned to NaN

- Add test to check that `specificity_rank` of `lrs_to_keep` is equal to min(specificity_rank)

0.1.4 (11.01.2023)
-----------------------------------------

- `rank_aggregate` will now sort interactions according to `magnitude_rank`.

- Fixed `SettingWithCopyWarning` warning when `return_all_lrs` is True

- Minor text improvements to the basic tutorial notebook

- Removed 'Print' from a verbose print message in `_choose_mtx_rep`


0.1.3 (07.12.2022)
-----------------------------------------
- Added `supp_columns` parameter to allow any column from liana to be returned.

- Added `return_all_lrs` parameter to allow all interactions to be returned with a `lrs_to_filter` flag for the interaction that do not pass the `expr_prop`, and each of those interactions is assigned to the worst **present** score from the ones that do pass the threshold.

- Fixed a bug where an exception was not thrown by `assert_covered`

- Raise explicit exceptions as text in multiple places.

- Changed cellphonedb p-values column name from "pvals" to "cellphone_pvals".

0.1.2
-----------------------------------------
- Added CellChat and GeometricMean methods

0.1.1
-----------------------------------------
- Add progress bar to permutations

- Deal with adata copies to optimize RAM

- change copy to inplace, and assign to uns, rather than return adata

- remove unnecessary filtering in _pre + extend units tests


0.1.0
-----------------------------------------
- Restructure API further

- Submit to PIP


0.0.3
-----------------------------------------
- Added a filter according to `min_cells` per cell identity

- prep_check_adata will now assert that `groupby` exists

- extended test_pre.py tests

- restructured the API to be more scverse-like

0.0.2
-----------------------------------------

- Added `dotplot` as a visualization option

- Added `basic_usage` tutorial

0.0.1
-----------------------------------------

First release alpha version of **liana-py**

- Re-implementations of:
    - CellPhoneDB

    - NATMI

    - SingleCellSignalR

    - Connectome

    - logFC

    - Robust aggregate rank

- Ligand-receptor resources as generated via OmniPathR.

