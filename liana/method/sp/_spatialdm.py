import numpy as np
from scipy.sparse import csr_matrix

from anndata import AnnData
from pandas import DataFrame
from typing import Optional

from liana.method.sp._SpatialMethod import _SpatialMeta
from liana.method.sp._spatial_pipe import _global_lr_pipe, _get_ordered_matrix
from liana.method.sp._spatial_utils import _local_to_dataframe, _standardize_matrix
from liana.method.sp._bivariate_funs import _global_spatialdm, _local_spatialdm


class SpatialDM(_SpatialMeta):
    def __init__(self, _method, _complex_cols, _obsm_keys):
        super().__init__(method_name=_method.method_name,
                         key_cols=_method.key_cols,
                         reference=_method.reference,
                         )

        self.complex_cols = _complex_cols
        self.obsm_keys = _obsm_keys
        self._method = _method

    def __call__(self,
                 adata: AnnData,
                 resource_name: str = 'consensus',
                 expr_prop: float = 0.05,
                 pvalue_method: str = 'analytical',
                 n_perm: int = 1000,
                 positive_only: bool = True,
                 use_raw: Optional[bool] = True,
                 layer: Optional[str] = None,
                 verbose: Optional[bool] = False,
                 seed: int = 1337,
                 resource: Optional[DataFrame] = None,
                 inplace=True
                 ):
        """
        Parameters
        ----------
        adata
            Annotated data object.
        resource_name
            Name of the resource to be loaded and use for ligand-receptor inference.
        expr_prop
            Minimum expression proportion for the ligands/receptors (and their subunits).
             Set to `0` to return unfiltered results.
        pvalue_method
            Method to obtain P-values: One out of ['permutation', 'analytical'];
            'analytical' by default.
        n_perm
            Number of permutations to be performed if `pvalue_method`=='permutation'
        positive_only
            Whether to calculate p-values only for positive correlations. `True` by default.
        use_raw
            Use raw attribute of adata if present.
        layer
            Layer in anndata.AnnData.layers to use. If None, use anndata.AnnData.X.
        verbose
            Verbosity flag
        seed
            Random seed for reproducibility.
        resource
            Parameter to enable external resources to be passed. Expects a pandas dataframe
            with [`ligand`, `receptor`] columns. None by default. If provided will overrule
            the resource requested via `resource_name`
        inplace
            If true return `DataFrame` with results, else assign to `.uns`.

        Returns
        -------
        If ``inplace = False``, returns:
        - 1) a `DataFrame` with ligand-receptor correlations for the whole slide (global)
        - 2) a `DataFrame` with ligand-receptor Moran's I for each spot
        - 3) a `DataFrame` with ligand-receptor correlations p-values for each spot
        Otherwise, modifies the ``adata`` object with the following keys:
        - :attr:`anndata.AnnData.uns` ``['global_res']`` with `1)`
        - :attr:`anndata.AnnData.obsm` ``['local_r']`` with  `2)`
        - :attr:`anndata.AnnData.obsm` ``['local_pvals']`` with  `3)`

        """
        if pvalue_method not in ['analytical', 'permutation']:
            raise ValueError('pvalue_method must be one of [analytical, permutation]')

        temp, lr_res, ligand_pos, receptor_pos = _global_lr_pipe(adata=adata,
                                                                 resource_name=resource_name,
                                                                 resource=resource,
                                                                 expr_prop=expr_prop,
                                                                 use_raw=use_raw,
                                                                 layer=layer,
                                                                 verbose=verbose,
                                                                 _key_cols=self.key_cols,
                                                                 _complex_cols=self.complex_cols,
                                                                 _obms_keys=self.obsm_keys
                                                                 )

        # n / sum(W) for Moran's I
        norm_factor = temp.obsm['proximity'].shape[0] / temp.obsm['proximity'].sum()
        dist = csr_matrix(norm_factor * temp.obsm['proximity'])
        
        x_key = 'ligand'
        y_key = 'receptor'
        # convert to spot_n x lr_n matrices
        x_mat = _get_ordered_matrix(mat=temp.X,
                                    pos=ligand_pos,
                                    order=lr_res[x_key])
        y_mat = _get_ordered_matrix(mat=temp.X,
                                    pos=receptor_pos,
                                    order=lr_res[y_key])

        # we use the same gene expression matrix for both x and y
        lr_res['global_r'], lr_res['global_pvals'] = \
            _global_spatialdm(x_mat=_standardize_matrix(x_mat, local=False, axis=1),
                              y_mat=_standardize_matrix(y_mat, local=False, axis=1),
                              dist=dist,
                              seed=seed,
                              n_perm=n_perm,
                              pvalue_method=pvalue_method,
                              positive_only=positive_only
                              )
        local_r, local_pvals = _local_spatialdm(x_mat=_standardize_matrix(x_mat, local=True, axis=1),
                                                y_mat=_standardize_matrix(y_mat, local=True, axis=1),
                                                dist=dist,  # TODO msq?
                                                seed=seed,
                                                n_perm=n_perm,
                                                pvalue_method=pvalue_method,
                                                positive_only=positive_only
                                                )

        # convert to dataframes
        local_r = _local_to_dataframe(array=local_r,
                                      idx=temp.obs.index,
                                      columns=lr_res['interaction'])
        local_pvals = _local_to_dataframe(array=local_pvals,
                                          idx=temp.obs.index,
                                          columns=lr_res['interaction'])

        if inplace:
            adata.uns['global_res'] = lr_res
            adata.obsm['local_r'] = local_r
            adata.obsm['local_pvals'] = local_pvals

        return None if inplace else (lr_res, local_r, local_pvals)



# initialize instance
_spatialdm = _SpatialMeta(
    method_name="SpatialDM",
    key_cols=['ligand_complex', 'receptor_complex'],
    reference="Zhuoxuan, L.I., Wang, T., Liu, P. and Huang, Y., 2022. SpatialDM: Rapid "
              "identification of spatially co-expressed ligand-receptor reveals cell-cell "
              "communication patterns. bioRxiv. "
)

spatialdm = SpatialDM(_method=_spatialdm,
                      _complex_cols=['ligand_means', 'receptor_means'],
                      _obsm_keys=['proximity']
                      )


