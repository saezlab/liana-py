Release notes
=============

0.1.3 (07.12.2022)
-----
- Added `supp_columns` parameter to allow any column of liana to be returned.
- Added `return_all_lrs` parameter to allow all interactions to be return, with
a `lrs_to_filter` flag for those that do not pass the `expr_prop`.
- Fixed a bug where an exception was not thrown by `assert_covered`
- Raise explicit exceptions as text in multiple places.

0.1.2
-----
- Added CellChat and GeometricMean methods

0.1.1
-----
- Add progress bar to permutations
- Deal with adata copies to optimize RAM
- change copy to inplace, and assign to uns, rather than return adata
- remove unnecessary filtering in _pre + extend units tests


0.1.0
-----
- Restructure API further
- Submit to PIP


0.0.3
-----
- Added a filter according to `min_cells` per cell identity
- prep_check_adata will now assert that `groupby` exists
- extended test_pre.py tests
- restructured the API to be more scverse-like

0.0.2
-----

- Added `dotplot` as a visualization option
- Added `basic_usage` tutorial

0.0.1
-----

First release alpha version of **liana-py**

- Re-implementations of:
    - CellPhoneDB
    - NATMI
    - SingleCellSignalR
    - Connectome
    - logFC
    - Robust aggregate rank

- Ligand-receptor resources as generated via OmniPathR.

