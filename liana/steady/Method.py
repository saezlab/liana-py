from .liana_pipe import liana_pipe
from anndata._core import anndata


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

    def __call__(self, adata: anndata, groupby: str, resource_name='consensus',
                 use_raw=False, layer=None, de_method='t-test', verbose=False,
                 n_perms=1000, seed=1337, resource=None):
        adata.uns['liana_res'] = liana_pipe(adata=adata,
                                            groupby=groupby,
                                            resource_name=resource_name,
                                            resource=resource,
                                            de_method=de_method,
                                            verbose=verbose,
                                            _score=self._SCORE,
                                            n_perms=n_perms,
                                            seed=seed,
                                            use_raw=use_raw,
                                            layer=layer,
                                            )
        return adata
