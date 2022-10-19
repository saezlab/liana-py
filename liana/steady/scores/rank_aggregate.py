from liana.steady.Method import MethodMeta
from liana.steady.liana_pipe import liana_pipe

from anndata import AnnData
from pandas import DataFrame
from typing import Optional


class ConsensusClass(MethodMeta):
    def __init__(self, _SCORE, methods):
        super().__init__(method_name=_SCORE.method_name,
                         complex_cols=[],
                         add_cols=[],
                         fun=_SCORE.fun,
                         magnitude=_SCORE.magnitude,
                         magnitude_desc=None,
                         specificity=_SCORE.specificity,
                         specificity_desc=None,
                         permute=_SCORE.permute,
                         reference=_SCORE.reference
                         )
        self._SCORE = _SCORE
        self.methods = methods
        self.steady = 'steady_rank'

        # Define scores to aggregate
        self.specificity_specs = {method.method_name: (
            method.specificity, method.specificity_desc) for method in methods
            if method.specificity is not None}
        self.magnitude_specs = {method.method_name: (
            method.magnitude, method.magnitude_desc) for method in methods
            if method.magnitude is not None}

        # If SingleCellSignalR is in there also add it to calculate steady
        methods_by_name = {method.method_name: method for method in methods}
        if 'SingleCellSignalR' in methods_by_name.keys():
            self.steady_specs = self.specificity_specs.copy()
            self.steady_specs['SingleCellSignalR'] = \
                (methods_by_name['SingleCellSignalR'].magnitude,
                 methods_by_name['SingleCellSignalR'].magnitude_desc)

        # Define additional columns needed depending on the methods to be run
        self.add_cols = list(
            {x for li in [method.add_cols for method in methods] for x in li}
        )
        self.complex_cols = list(
            {x for li in [method.complex_cols for method in methods] for x in li}
        )

    def __call__(self,
                 adata: AnnData,
                 groupby: str,
                 resource_name: str = 'consensus',
                 expr_prop: float = 0.1,
                 base: float = 2.718281828459045,
                 aggregate_method='rra',
                 consensus_opts=None,
                 use_raw: Optional[bool] = False,
                 layer: Optional[str] = None,
                 de_method='t-test',
                 verbose: Optional[bool] = False,
                 n_perms: int = 1000,
                 seed: int = 1337,
                 resource: Optional[DataFrame] = None) -> AnnData:
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
        base
            Exponent base used to reverse the log-transformation of matrix. Note that this is
            relevant only for the `logfc` method.
        use_raw
            Use raw attribute of adata if present.
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

        Returns
        -------
        If ``copy = True``, returns a `DataFrame` with ligand-receptor results
        Otherwise, modifies the ``adata`` object with the following key:
            - :attr:`anndata.AnnData.uns` ``['liana_res']`` with the aforementioned DataFrame
        """
        adata.uns['liana_res'] = liana_pipe(adata=adata,
                                            groupby=groupby,
                                            resource_name=resource_name,
                                            resource=resource,
                                            expr_prop=expr_prop,
                                            supp_cols=None,  # None for now - subunit ambiguity
                                            base=base,
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
        return adata


_consensus_meta = MethodMeta(method_name="Consensus",
                             complex_cols=[],
                             add_cols=[],
                             fun=None,  # change to _robust_rank
                             magnitude='magnitude_rank',
                             magnitude_desc=False,
                             specificity='specificity_rank',
                             specificity_desc=False,
                             permute=False,
                             reference='Kolde, R., Laur, S., Adler, P. and Vilo, J., 2012. Robust '
                                       'rank aggregation for gene list integration and '
                                       'meta-analysis. Bioinformatics, 28(4), pp.573-580. '
                             )
