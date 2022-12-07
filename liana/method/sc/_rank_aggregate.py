from liana.method._Method import MethodMeta
from liana.method._liana_pipe import liana_pipe

from anndata import AnnData
from pandas import DataFrame
from typing import Optional


class AggregateClass(MethodMeta):
    """LIANA's Method Consensus Class"""
    def __init__(self, _SCORE, methods):
        super().__init__(method_name=_SCORE.method_name,
                         complex_cols=[],
                         add_cols=[],
                         fun=_SCORE.fun,
                         magnitude=_SCORE.magnitude,
                         magnitude_ascending=None,
                         specificity=_SCORE.specificity,
                         specificity_ascending=None,
                         permute=_SCORE.permute,
                         reference=_SCORE.reference
                         )
        self._SCORE = _SCORE
        self.methods = methods
        self.steady = 'steady_rank'
        self.steady_ascending = True

        # Define sc to aggregate
        self.specificity_specs = {method.method_name: (
            method.specificity, method.specificity_ascending) for method in methods
            if method.specificity is not None}
        self.magnitude_specs = {method.method_name: (
            method.magnitude, method.magnitude_ascending) for method in methods
            if method.magnitude is not None}

        # If SingleCellSignalR is in there also add it to calculate method
        methods_by_name = {method.method_name: method for method in methods}
        if 'SingleCellSignalR' in methods_by_name.keys():
            self.steady_specs = self.specificity_specs.copy()
            self.steady_specs['SingleCellSignalR'] = \
                (methods_by_name['SingleCellSignalR'].magnitude,
                 methods_by_name['SingleCellSignalR'].magnitude_ascending)

        # Define additional columns needed depending on the methods to be run
        self.add_cols = list(
            {x for li in [method.add_cols for method in methods] for x in li}
        )
        self.complex_cols = list(
            {x for li in [method.complex_cols for method in methods] for x in li}
        )

    def describe(self):
        """Briefly described the method"""
        print(
            f"{self.method_name} returns `{self.magnitude}`, `{self.specificity}`,"
            f" and `{self.steady}`. "
            f"{self.magnitude} and {self.specificity} represent an aggregate of the "
            f"`magnitude`- and `specificity`-related scoring functions from the different methods."
            f"{self.steady} represents one scoring function from each method intended"
            f" to prioritize ligand-receptor interactions in steady-state data, "
            f"regardless if they represent `specificity` or `magnitude`."
        )

    def __call__(self,
                 adata: AnnData,
                 groupby: str,
                 resource_name: str = 'consensus',
                 expr_prop: float = 0.1,
                 min_cells: int = 5,
                 base: float = 2.718281828459045,
                 aggregate_method='rra',
                 return_all_lrs: bool = False,
                 consensus_opts=None,
                 use_raw: Optional[bool] = True,
                 layer: Optional[str] = None,
                 de_method='t-test',
                 verbose: Optional[bool] = False,
                 n_perms: int = 1000,
                 seed: int = 1337,
                 resource: Optional[DataFrame] = None,
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
        aggregate_method
            Method aggregation approach, one of ['mean', 'rra'], where `mean` represents the
            mean rank, while 'rra' is the RobustRankAggregate (Kolde et al., 2014)
            of the interactions
        return_all_lrs
            Bool whether to return all LRs, or only those that surpass the `expr_prop`
            threshold. Those interactions that do not pass the `expr_prop` threshold will
            be assigned to the *worst* score of the ones that do. `False` by default.
        use_raw
            Use raw attribute of adata if present. True, by default.
        layer
            Layer in anndata.AnnData.layers to use. If None, use anndata.AnnData.X.
        de_method
            Differential expression method. `scanpy.tl.rank_genes_groups` is used to rank genes
            according to 1vsRest. The default method is 't-test'.
        verbose
            Verbosity flag
        n_perms
            Number of permutations for the permutation test. Note that this is relevant
            only for permutation-based methods - e.g. `CellPhoneDB`
        seed
            Random seed for reproducibility.
        resource
            Parameter to enable external resources to be passed. Expects a pandas dataframe
            with [`ligand`, `receptor`] columns. None by default. If provided will overrule
            the resource requested via `resource_name`
        inplace
            If true return `DataFrame` with results, else assign inplace to `.uns`.


        Returns
        -------
        If ``inplace = False``, returns a `DataFrame` with ligand-receptor results
        Otherwise, modifies the ``adata`` object with the following key:
            - :attr:`anndata.AnnData.uns` ``['liana_res']`` with the aforementioned DataFrame
        """
        liana_res = liana_pipe(adata=adata,
                               groupby=groupby,
                               resource_name=resource_name,
                               resource=resource,
                               expr_prop=expr_prop,
                               min_cells=min_cells,
                               base=base,
                               return_all_lrs=return_all_lrs,
                               de_method=de_method,
                               verbose=verbose,
                               _score=self,
                               use_raw=use_raw,
                               layer=layer,
                               n_perms=n_perms,
                               seed=seed,
                               _methods=self.methods,
                               _aggregate_method=aggregate_method,
                               _consensus_opts=consensus_opts
                               )
        adata.uns['liana_res'] = liana_res

        return None if inplace else liana_res


_rank_aggregate_meta = \
    MethodMeta(method_name="Rank_Aggregate",
               complex_cols=[],
               add_cols=[],
               fun=None,  # change to _robust_rank
               magnitude='magnitude_rank',
               magnitude_ascending=True,
               specificity='specificity_rank',
               specificity_ascending=True,
               permute=False,
               reference='Dimitrov, D., Türei, D., Garrido-Rodriguez, M., Burmedi, P.L., '
                         'Nagai, J.S., Boys, C., Ramirez Flores, R.O., Kim, H., Szalai, B., '
                         'Costa, I.G. and Valdeolivas, A., 2022. Comparison of methods and '
                         'resources for cell-cell communication inference from single-cell '
                         'RNA-Seq data. Nature Communications, 13(1), pp.1-13. '
               )
