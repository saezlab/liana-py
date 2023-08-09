"""
Preprocessing functions.
Functions to preprocess the anndata object prior to running any method.
"""
from __future__ import annotations

import numpy as np
from anndata import AnnData
from typing import Optional
from pandas import DataFrame, Index
import scanpy as sc
from scipy.sparse import csr_matrix, isspmatrix_csr


def assert_covered(
        subset,
        superset,
        subset_name: str = "resource",
        superset_name: str = "var_names",
        prop_missing_allowed: float = 0.98,
        verbose: bool = False) -> None:
    """
    Assert if elements are covered at a decent proportion

    Parameters
    ----------
    subset
        Subset of elements
    superset
        The superset of elements
    subset_name
        Name of the subset
    superset_name
        Name of the superset
    prop_missing_allowed
        Allowed proportion of missing/mismatched elements in the subset
    verbose
        Verbosity flag

    Returns
    -------
    None
    """

    subset = np.asarray(subset)
    is_missing = ~np.isin(subset, superset)
    prop_missing = np.sum(is_missing) / len(subset)
    x_missing = ", ".join([x for x in subset[is_missing]])

    if prop_missing > prop_missing_allowed:
        msg = (
            f"Please check if appropriate organism/ID type was provided! "
            f"Allowed proportion ({prop_missing_allowed}) of missing "
            f"{subset_name} elements exceeded ({prop_missing:.2f}). "
            f"Too few features from the resource were found in the data."
        )
        raise ValueError(msg + f" [{x_missing}] missing from {superset_name}")

    if verbose & (prop_missing > 0):
        print(f"{prop_missing:.2f} of entities in the resource are missing from the data.")


def prep_check_adata(adata: AnnData,
                     groupby: (str | None),
                     min_cells: (int | None),
                     use_raw: Optional[bool] = False,
                     layer: Optional[str] = None,
                     obsm = None,
                     complex_sep='_',
                     verbose: Optional[bool] = False) -> AnnData:
    """
    Check if the anndata object is in the correct format and preprocess

    Parameters
    ----------
    adata
        Un-formatted Anndata.
    groupby
        column to groupby. None if the ligand-receptor pipe
        calling this function does not rely on cell labels.
        For example, if ligand-receptor stats are needed
        for the whole sample (global).
    min_cells
        minimum cells per cell identity. None if groupby is not passed.
    use_raw
        Use raw attribute of adata if present.
    layer
        Indicate whether to use any layer.
    verbose
        Verbosity flag.

    Returns
    -------
    Anndata object to be used downstream
    """
    X = _choose_mtx_rep(adata=adata, use_raw=use_raw,
                        layer=layer, verbose=verbose)

    if use_raw & (layer is None):
        var = DataFrame(index=adata.raw.var_names)
    else:
        var = DataFrame(index=adata.var_names)

    adata = sc.AnnData(X=X,
                       obs=adata.obs.copy(),
                       dtype="float32",
                       var=var,
                       obsp=adata.obsp.copy(),
                       obsm=obsm
                       ).copy()
    adata.var_names_make_unique()

    # Check for empty features
    msk_features = np.sum(adata.X, axis=0).A1 == 0
    n_empty_features = np.sum(msk_features)
    if n_empty_features > 0:
        if verbose:
            print(f"{n_empty_features} features of mat are empty, they will be removed.")
        adata = adata[:, ~msk_features]

    # Check for empty samples
    msk_samples = adata.X.sum(axis=1).A1 == 0
    n_empty_samples = np.sum(msk_samples)
    if n_empty_samples > 0:
        print(f"{n_empty_samples} samples of mat are empty, please remove them.")

    # Check if log-norm
    _sum = np.sum(adata.X.data[0:100])
    if _sum == np.floor(_sum):
        if verbose:
            print("Make sure that normalized counts are passed!")

    # Check for non-finite values
    if np.any(~np.isfinite(adata.X.data)):
        raise ValueError(
            "mat contains non finite values (nan or inf), please set them to 0 or remove them.")

    # Define idents col name
    if groupby is not None:
        if groupby not in adata.obs.columns:
            raise AssertionError(f"`{groupby}` not found in `adata.obs.columns`.")
        adata.obs.loc[:, 'label'] = adata.obs[groupby]

        # Remove any cell types below X number of cells per cell type
        count_cells = adata.obs.groupby(groupby)[groupby].size().reset_index(name='count').copy()
        count_cells['keep'] = count_cells['count'] >= min_cells

        if not all(count_cells.keep):
            lowly_abundant_idents = list(count_cells[~count_cells.keep][groupby])
            # remove lowly abundant identities
            msk = ~np.isin(adata.obs[[groupby]], lowly_abundant_idents)
            adata = adata[msk]
            if verbose:
                print("The following cell identities were excluded: {0}".format(
                    ", ".join(lowly_abundant_idents)))
                
    check_vars(adata.var_names,
               complex_sep=complex_sep,
               verbose=verbose)

    # Re-order adata vars alphabetically
    adata = adata[:, np.sort(adata.var_names)]

    return adata


# format variable names
def check_vars(var_names, complex_sep, verbose=False) -> list:
    """
    Raise a warning if `complex_sep` is part of any variable name.
    """ 
    var_issues = []
    if complex_sep is not None:
        for name in var_names:
            if complex_sep in name:
                var_issues.append(name)
    else:
        pass
    
    if verbose & (len(var_issues) > 0):
        print(f"Warning: {var_issues} contain `{complex_sep}`. Consider replacing those!")



def filter_resource(resource: DataFrame, var_names: Index) -> DataFrame:
    """
    Filter interactions for which vars are not present.

    Note that here I remove any interaction that /w genes that are not found
    in the dataset. Note that this is not necessarily the case in liana-r.
    There, I assign the expression of those with missing subunits to 0, while
    those without any subunit present are implicitly filtered.

    Parameters
    ---------
    resource
        Resource with 'ligand' and 'receptor' columns
    var_names
        Relevant variables - i.e. the variables to be used downstream

    Returns
    ------
    A filtered resource dataframe
    """
    # Remove those without any subunit
    resource = resource[(np.isin(resource.ligand, var_names)) &
                        (np.isin(resource.receptor, var_names))]

    # Only keep interactions /w complexes for which all subunits are present
    missing_comps = resource[resource.interaction.str.contains('_')].copy()
    missing_comps['all_units'] = \
        missing_comps['ligand_complex'] + '_' + missing_comps[
            'receptor_complex']

    # Get those not with all subunits
    missing_comps = missing_comps[np.logical_not(
        [all([x in var_names for x in entity.split('_')])
         for entity in missing_comps.all_units]
    )]
    # Filter them
    return resource[~resource.interaction.isin(missing_comps.interaction)]


def _choose_mtx_rep(adata, use_raw=False, layer=None, verbose=False) -> csr_matrix:
    """
    Choose matrix (adapted from scanpy)
    
    Parameters
    ----------
    adata
        Unformatted Anndata.
    use_raw
        Use raw attribute of adata if present.
    layer
        Indicate whether to use any layer.
    
    Returns
    -------
        The matrix to be used by liana-py.
    """
    is_layer = layer is not None
    if is_layer & use_raw:
        raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
    if is_layer:
        if verbose:
            print(f"Using the `{layer}` layer!")
        X = adata.layers[layer]
    elif use_raw:
        if adata.raw is None:
            raise ValueError("`.raw` is not initialized!")
        if verbose:
            print("Using `.raw`!")
        X = adata.raw.X
    else:
        if verbose:
            print("Using `.X`!")
        X = adata.X
        
    # convert to sparse csr matrix
    if not isspmatrix_csr(X):
        if verbose:
            print("Converting mat to CSR format")
        X = csr_matrix(X)
    
    return X
