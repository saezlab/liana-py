from liana.steady.Method import MethodMeta
from liana.steady.liana_pipe import liana_pipe
from liana import cellphonedb, natmi, connectome, logfc, singlecellsignalr as sca

from anndata._core import anndata

_consensus_meta = MethodMeta(method_name="Consensus",
                             complex_cols=[],
                             add_cols=[],
                             fun=None,  # change to _robust_rank
                             magnitude='rra_magnitude',
                             magnitude_desc=False,
                             specificity='rra_specificity',
                             specificity_desc=False,
                             permute=False,
                             reference=''
                             )


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
        # Define additional columns needed depending on the methods to be run
        self.add_cols = list(
            {x for li in [method.add_cols for method in methods] for x in li}
        )
        self.complex_cols = list(
            {x for li in [method.complex_cols for method in methods] for x in li}
        )

    def __call__(self, adata: anndata, groupby: str, resource_name='consensus',
                 de_method='wilcoxon', verbose=False, n_perms=1000, seed=1337,
                 resource=None):
        adata.uns['liana_res'] = liana_pipe(adata=adata,
                                            groupby=groupby,
                                            resource_name=resource_name,
                                            resource=resource,
                                            de_method=de_method,
                                            verbose=verbose,
                                            _score=self,
                                            n_perms=n_perms,
                                            seed=seed,
                                            _methods=self.methods
                                            )
        return adata


# Create callable consensus object
_methods = [cellphonedb, connectome, logfc, natmi, sca]
consensus = ConsensusClass(_consensus_meta, methods=_methods)
