
import scanpy as sc
import muon as mu
import anndata as an

from liana._constants import DefaultValues as V

from liana._docs import d
from liana._logging import _logg


@d.dedent
def anndata_to_mdata(
    adata: an.AnnData,
    groupby: str,
    min_cells = V.min_cells,
    lr_sep = V.lr_sep,
    by_var: bool = False,
    verbose = V.verbose,

):
    """
    Convert AnnData object to MultiData object.

    Args:
    ----------
    %(adata)s
    %(groupby)s
    %(min_cells)s
    by_var: bool, optional
        Whether to group by variables instead of observations. Default is False.
    %(lr_sep)s
    %(verbose)s

    Returns
    -------
        MultiData: A new MultiData object.
    """

    adata_dict = {}

    if by_var:
        # Extract modalities from variable name prefixes before the `lr_sep` separator
        modalities = adata.var.index.str.split(lr_sep, expand=True).get_level_values(0).unique().tolist()

        for modality in modalities:
            if verbose:
                _logg(f"Processing modality (from var): {modality}")

            # Filter variables (genes) belonging to the current modality
            var_filter = adata.var.index.str.startswith(f"{modality}{lr_sep}")
            adata_mod = adata[:, var_filter].copy()

            # Apply gene filtering within this modality
            sc.pp.filter_genes(adata_mod, min_cells=min_cells)

            adata_dict[modality] = adata_mod
            if verbose:
                _logg(f"Modality {modality} (from var): {adata_mod.n_obs} cells, {adata_mod.n_vars} genes after filtering.")

    else:
        # Extract modalities from the observation annotation based on `groupby`
        modalities = adata.obs[groupby].unique().tolist()

        for modality in modalities:
            if verbose:
                _logg(f"Processing modality (from obs): {modality}")

            # Filter observations (cells) belonging to the current modality
            adata_mod = adata[adata.obs[groupby] == modality].copy()

            # Apply gene filtering
            sc.pp.filter_genes(adata_mod, min_cells=min_cells)


            adata_dict[modality] = adata_mod
            if verbose:
                _logg(f"Modality {modality} (from obs): {adata_mod.n_obs} cells, {adata_mod.n_vars} genes after filtering.")

    # Create and return a MuData object with filtered modalities
    mudata = mu.MuData(adata_dict)
    return mudata
        