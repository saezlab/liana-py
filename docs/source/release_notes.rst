Release notes
=============

0.1.5 (11.01.2023)
-----
- Hotfix `return_all_lrs` specificity_rank being assigned to NaN
- Add test to check that `specificity_rank` of `lrs_to_keep` is equal to min(specificity_rank)

0.1.4 (11.01.2023)
-----

- `rank_aggregate` will now sort interactions according to `magnitude_rank`.
- Fixed `SettingWithCopyWarning` warning when `return_all_lrs` is True
- Minor text improvements to the basic tutorial notebook
- Removed 'Print' from a verbose print message in `_choose_mtx_rep`


0.1.3 (07.12.2022)
-----
- Added `supp_columns` parameter to allow any column from liana to be returned.
- Added `return_all_lrs` parameter to allow all interactions to be returned with a `lrs_to_filter` flag for the interaction that do not pass the `expr_prop`, and each of those interactions is assigned to the worst **present** score from the ones that do pass the threshold.
- Fixed a bug where an exception was not thrown by `assert_covered`
- Raise explicit exceptions as text in multiple places.
- Changed cellphonedb p-values column name from "pvals" to "cellphone_pvals".

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

