from __future__ import annotations
from liana.method._Method import MethodMeta

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
                 method_name: str,
                 fun,
                 reference: str
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
        self.method_name = method_name
        self.fun = fun
        self.reference = reference

    # describe self
    def describe(self):
        """Briefly described the method"""
        print(
            f"{self.method_name} uses xxx"
            f" xx"
        )

    def reference(self):
        """Prints out reference in Harvard format"""
        print(self.reference)

    def get_meta(self):
        """Returns method metadata as pandas row"""
        meta = DataFrame([{"Method Name": self.method_name,
                           "Reference": self.reference
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
    def __init__(self, _ESTIMATION):
        super().__init__(method_name=_ESTIMATION.method_name,
                         fun=_ESTIMATION.fun,
                         reference=_ESTIMATION.reference
                         )
        self._ESTIMATION = _ESTIMATION

    def __call__(self,
                 adata: AnnData,
                 groupby: str,
                 resource_name: str = 'consensus',
                 resource: Optional[DataFrame] = None,
                 met_est_resource_name: str = 'consensus',
                 met_est_resource: Optional[DataFrame] = None,
                 expr_prop: float = 0.1,
                 min_cells: int = 5,
                 base: float = 2.718281828459045,
                 supp_columns: list = None,
                 return_all_lrs: bool = False,
                 use_raw: Optional[bool] = True,
                 layer: Optional[str] = None,
                 de_method='t-test',
                 verbose: Optional[bool] = False,
                 inplace=True):
        """
        Parameters
        ----------
        adata
            Annotated data object.
        groupby
            The key of the observations grouping to consider.
        resource_name
            Name of the resource to be loaded and use for ligand-receptor inference.
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
                               resource_name='mebocost',
                               resource=resource,
                               met_est_resource_name=met_est_resource_name,
                               met_est_resource=met_est_resource,
                               expr_prop=expr_prop,
                               min_cells=min_cells,
                               supp_columns=supp_columns,
                               return_all_lrs=return_all_lrs,
                               base=base,
                               de_method=de_method,
                               verbose=verbose,
                               _estimation=self._ESTIMATION,
                               use_raw=use_raw,
                               layer=layer,
                               )
        if inplace:
            adata.obsm['metabolite_abundance'] = ml_res

        
        return None if inplace else ml_res
        
        
    


def _show_met_est_methods(metab_methods):
    return concat([mmethod.get_meta() for mmethod in metab_methods])



