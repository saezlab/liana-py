from __future__ import annotations

from ._ml_pipe import ml_pipe

from anndata import AnnData
from pandas import DataFrame, concat
from typing import Optional
import weakref


class MetabMethodMeta:
    """
    A Class used to store Method Metadata
    """
    
    # initiate a list to store weak references to all instances
    instances = [] ## TODO separate instances for each subclass
    
    def __init__(self,
                 score_method_name: str,
                 fun,
                 est_reference: str,
                 score_reference: str,
                 complex_cols: list,
                 add_cols: list,
                 magnitude: str | None,
                 magnitude_ascending: bool | None,
                 specificity: str | None,
                 specificity_ascending: bool | None,
                 permute: bool,
                 agg_fun,
                 ):
        """
        Parameters
        ----------
        method_name
            Name of the Method
        fun
            Interaction Scoring function
        reference
            Publication reference in Harvard style
        """
        self.__class__.instances.append(weakref.proxy(self))
        self.score_method_name = score_method_name
        self.est_reference = est_reference
        self.complex_cols = complex_cols
        self.add_cols = add_cols
        self.fun = fun
        self.magnitude = magnitude
        self.magnitude_ascending = magnitude_ascending
        self.specificity = specificity
        self.specificity_ascending = specificity_ascending
        self.permute = permute
        self.score_reference = score_reference
        self.agg_fun = agg_fun

    # describe self
    def describe(self):
        """Briefly described the method"""
        print(
            f"{self.score_method_name} uses xxx"
            f" xx"
        )

    def score_reference(self):
        """Prints out reference in Harvard format"""
        print(self.score_reference)

    def get_meta(self):
        """Returns method metadata as pandas row"""
        meta = DataFrame([{"Method Name": self.score_method_name,
                           "Reference": self.score_reference
                           }])
        return meta
    
    def by_sample(self, adata, sample_key, inplace=True, verbose=False, **kwargs):
        """
        Run a method by sample.
        
        Parameters
        ----------
            adata 
                AnnData object to run the method on
            
            sample_key
                key in `adata.obs` to use for grouping by sample/context
                
            inplace
                whether to store the results in `adata.uns['liana_res']` or return a dataframe
            
            verbose
                whether to print verbose output
            
            **kwargs
                keyword arguments to pass to the method
        
        Returns
        -------
        A pandas DataFrame with the results and a column sample if inplace is False, else None
        
        """
        
        if sample_key not in adata.obs:
            raise ValueError(f"{sample_key} was not found in `adata.obs`.")

        
        if inplace:
            adata.obsm['metabolite_abundance'] = adata.X
        return None if inplace else adata.X


class MetabMethod(MetabMethodMeta):
    """
    liana's Method Class
    """
    def __init__(self, _SCORE):
        super().__init__(score_method_name=_SCORE.score_method_name,
                         est_reference=_SCORE.est_reference,
                         score_reference=_SCORE.score_reference,
                         complex_cols=_SCORE.complex_cols,
                         add_cols=_SCORE.add_cols,
                         fun=_SCORE.fun,
                         magnitude=_SCORE.magnitude,
                         magnitude_ascending=_SCORE.magnitude_ascending,
                         specificity=_SCORE.specificity,
                         specificity_ascending=_SCORE.specificity_ascending,
                         permute=_SCORE.permute,
                         agg_fun=_SCORE.agg_fun
                         )
        self._SCORE = _SCORE
        


    def __call__(self,
                 adata: AnnData,
                 groupby: str,
                 resource: Optional[DataFrame] = None,
                 resource_name: str = 'metalinksdb',
                 met_est_resource_name: str = 'metalinksdb',
                 met_est_resource: Optional[DataFrame] = None,
                 est_fun: str = 'mean_per_cell',
                 expr_prop: float = 0.1,
                 min_cells: int = 5,
                 base: float = 2.718281828459045,
                 supp_columns: list = None,
                 return_all_lrs: bool = False,
                 use_raw: Optional[bool] = True,
                 layer: Optional[str] = None,
                 verbose: Optional[bool] = False,
                 n_perms: int = 1000,
                 seed: int = 1337,
                 inplace=True):
        """
        Parameters
        ----------
        adata
            Annotated data object.
        groupby
            The key of the observations grouping to consider.
        output
            Full MR calculation (CCC) or only metabolite estimation (ME).
        resource_name
            Name of the resource to be loaded and use for ligand-receptor inference.
        met_est_resource_name
            Name of the resource to be loaded and use for metabolite estimation.
        met_est_resource
            Metabolite-gene links to be used for metabolite estimation.
        expr_prop
            Minimum expression proportion for the ligands/receptors (and their subunits) in the
            corresponding cell identities. Set to `0`, to return unfiltered results.
        min_cells
            Minimum cells per cell identity (`groupby`) to be considered for downstream analysis
        base
            Exponent base used to reverse the log-transformation of matrix. Note that this is
            relevant only for the `logfc` method.
        supp_columns
            Any additional columns to be added from any of the methods implemented in
            liana, or any of the columns returned by `scanpy.tl.rank_genes_groups`, each
            starting with ligand_* or receptor_*. For example, `['ligand_pvals', 'receptor_pvals']`.
            `None` by default.
        return_all_lrs
            Bool whether to return all LRs, or only those that surpass the `expr_prop`
            threshold. Those interactions that do not pass the `expr_prop` threshold will
            be assigned to the *worst* score of the ones that do. `False` by default.
        use_raw
            Use raw attribute of adata if present.
        layer
            Layer in anndata.AnnData.layers to use. If None, use anndata.AnnData.X.
        de_method
            Differential expression method. `scanpy.tl.rank_genes_groups` is used to rank genes
            according to 1vsRest. The default method is 't-test'. Only relevant if p-values
            are included in `supp_cols`
        verbose
            Verbosity flag
        resource
            Parameter to enable external resources to be passed. Expects a pandas dataframe
            with [`ligand`, `receptor`] columns. None by default. If provided will overrule
            the resource requested via `resource_name`
        inplace
            If true return `DataFrame` with results, else assign to `.layer`.

        Returns
        -------
            If ``inplace = False``, returns a `DataFrame` with ligand-receptor results
            Otherwise, modifies the ``adata`` object with the following key:
            - :attr:`anndata.AnnData.uns` ``['liana_res']`` with the aforementioned DataFrame
        """
        if supp_columns is None:
            supp_columns = []

        ml_res = ml_pipe(adata=adata,
                        groupby=groupby,
                        resource_name=resource_name,
                        resource=resource,
                        met_est_resource_name=met_est_resource_name,
                        met_est_resource=met_est_resource,
                        expr_prop=expr_prop,
                        est_fun=est_fun,
                        min_cells=min_cells,
                        supp_columns=supp_columns,
                        return_all_lrs=return_all_lrs,
                        verbose=verbose,
                        use_raw=use_raw,
                        n_perms=n_perms,
                        seed=seed,
                        layer=layer,
                        _score = self._SCORE,)
        if inplace:
            adata.uns['CCC_res'] = ml_res[0]
            adata.obsm['metabolite_abundance'] = ml_res[1]
            adata.uns['mask'] = ml_res[2]
            # adata.uns['met_meta'] = ml_res[3]

        
        return None if inplace else ml_res
        
