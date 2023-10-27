from __future__ import annotations

from liana.method.sc._Method import MethodMeta
from liana.method.sc._liana_pipe import liana_pipe
from liana._constants._docs import d
from liana.utils import mdata_to_anndata
from mudata import MuData

import anndata as an
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
                         magnitude_ascending=True,
                         specificity=_SCORE.specificity,
                         specificity_ascending=True,
                         permute=_SCORE.permute,
                         reference=_SCORE.reference
                         )
        self._SCORE = _SCORE
        self.methods = methods

        # Define sc to aggregate
        self.specificity_specs = {method.method_name: (
            method.specificity, method.specificity_ascending) for method in methods
            if method.specificity is not None}
        self.magnitude_specs = {method.method_name: (
            method.magnitude, method.magnitude_ascending) for method in methods
            if method.magnitude is not None}

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
            f"{self.method_name} returns `{self.magnitude}`, `{self.specificity}`. "
            f"{self.magnitude} and {self.specificity} respectively represent an aggregate of the "
            f"`magnitude`- and `specificity`-related scoring functions from the different methods."
        )

    @d.dedent
    def __call__(self,
                 adata: an.AnnData | MuData,
                 groupby: str,
                 resource_name: str = 'consensus',
                 expr_prop: float = 0.1,
                 min_cells: int = 5,
                 base: float = 2.718281828459045,
                 aggregate_method='rra',
                 return_all_lrs: bool = False,
                 key_added : str = 'liana_res',
                 consensus_opts=None,
                 use_raw: Optional[bool] = True,
                 layer: Optional[str] = None,
                 de_method='t-test',
                 verbose: Optional[bool] = False,
                 n_perms: int | None = 1000 ,
                 seed: int = 1337,
                 resource: Optional[DataFrame] = None,
                 interactions=None,
                 mdata_kwargs = dict(),
                 inplace=True
                 ):
        """
        Get an aggregate of ligand-receptor scores from multiple methods.
        
        Parameters
        ----------
        %(adata)s
        %(groupby)s
        %(resource_name)s
        %(expr_prop)s
        %(min_cells)s
        %(base)s
        aggregate_method
            Method aggregation approach, one of ['mean', 'rra'], where `mean` represents the
            mean rank, while 'rra' is the RobustRankAggregate (Kolde et al., 2014)
            of the interactions
        %(return_all_lrs)s
        %(key_added)s
        %(use_raw)s
        %(layer)s
        %(de_method)s
        %(verbose)s
        %(n_perms_sc)s
        %(seed)s
        %(resource)s
        %(interactions)s
        %(mdata_kwargs)s
        %(inplace)s

        Returns
        -------
        If ``inplace = False``, returns a `DataFrame` with ligand-receptor results
        Otherwise, modifies the ``adata`` object with the following key:
            - :attr:`anndata.AnnData.uns` ``['liana_res']`` with the aforementioned DataFrame
        """
        
        if isinstance(adata, MuData):
            ad = mdata_to_anndata(adata, **mdata_kwargs, verbose=verbose)
        else:
            ad = adata
        
        liana_res = liana_pipe(adata=ad,
                               groupby=groupby,
                               resource_name=resource_name,
                               resource=resource,
                               interactions=interactions,
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
        
        if inplace:
            adata.uns[key_added] = liana_res
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
