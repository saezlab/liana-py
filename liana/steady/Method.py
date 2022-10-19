from .liana_pipe import liana_pipe

from anndata import AnnData
from pandas import DataFrame
from typing import Optional


# Object to store MetaData
class MethodMeta:
    def __init__(self, method_name, complex_cols, add_cols, fun, magnitude,
                 magnitude_desc, specificity, specificity_desc, permute,
                 reference):
        self.method_name = method_name  # method name
        self.complex_cols = complex_cols  # complex-relevant columns
        self.add_cols = add_cols  # additional columns
        self.fun = fun  # Function to run
        self.magnitude = magnitude  # Name of the col
        self.magnitude_desc = magnitude_desc  # desc or not
        self.specificity = specificity  # Name of the col
        self.specificity_desc = specificity_desc  # desc or not
        self.permute = permute  # True/False
        self.reference = reference  # Publication

    # describe self
    def describe(self):
        print(
            f"{self.method_name} uses `{self.magnitude}` and `{self.specificity}`"
            f" as measures of expression strength and interaction specificity, respectively"
        )

    def reference(self):
        print(self.reference)


# Class To initialize Method objects. Will likely need to create a specific
# class for each method in order to allow redundant parameters to be removed
# (e.g. `de_method` for cpdb)
class Method(MethodMeta):
    def __init__(self, _SCORE):
        super().__init__(method_name=_SCORE.method_name,
                         complex_cols=_SCORE.complex_cols,
                         add_cols=_SCORE.add_cols,
                         fun=_SCORE.fun,
                         magnitude=_SCORE.magnitude,
                         magnitude_desc=_SCORE.magnitude_desc,
                         specificity=_SCORE.specificity,
                         specificity_desc=_SCORE.specificity_desc,
                         permute=_SCORE.permute,
                         reference=_SCORE.reference
                         )
        self._SCORE = _SCORE

    def __call__(self,
                 adata: AnnData,
                 groupby: str,
                 resource_name: str = 'consensus',
                 expr_prop: float = 0.1,
                 base: float = 2.718281828459045,
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
        :returns:
        If ``copy = True``, returns a `DataFrame` with ligand-receptor results
        Otherwise, modifies the ``adata`` object with the following key:
            - :attr:`anndata.AnnData.uns` ``['liana_res']`` with the aforementioned DataFrame
        """
        adata.uns['liana_res'] = liana_pipe(adata=adata,
                                            groupby=groupby,
                                            resource_name=resource_name,
                                            resource=resource,
                                            expr_prop=expr_prop,
                                            supp_cols=['ligand_pvals', 'receptor_pvals'],
                                            base=base,
                                            de_method=de_method,
                                            verbose=verbose,
                                            _score=self._SCORE,
                                            n_perms=n_perms,
                                            seed=seed,
                                            use_raw=use_raw,
                                            layer=layer,
                                            )
        return adata

"""

If ``copy = True``, returns a :class:`dict` with following keys:
    - `'means'` - :class:`pandas.DataFrame` containing the mean expression.
    - `'pvalues'` - :class:`pandas.DataFrame` containing the possibly corrected p-values.
    - `'metadata'` - :class:`pandas.DataFrame` containing interaction metadata.
Otherwise, modifies the ``adata`` object with the following key:
    - :attr:`anndata.AnnData.uns` ``['{key_added}']`` - the above mentioned :class:`dict`.
`NaN` p-values mark combinations for which the mean expression of one of the interacting components was 0
or it didn't pass the ``threshold`` percentage of cells being expressed within a given cluster.

"""