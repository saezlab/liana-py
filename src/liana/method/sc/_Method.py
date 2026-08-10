from __future__ import annotations

import weakref
from collections.abc import Callable

import anndata as an
from mudata import MuData
from pandas import DataFrame, concat
from tqdm import tqdm

from liana._constants import DefaultValues as V
from liana._constants import Keys as K
from liana._docs import d
from liana._logging import _logg
from liana.method.sc._liana_pipe import liana_pipe


class MethodMeta:
    """
    A Class used to store Method Metadata

    Parameters
    ----------
    method_name
        Name of the Method
    complex_cols
        Columns relevant for protein complexes
    add_cols
        Additional columns required by the method
    fun
        Interaction Scoring function
    magnitude
        Name of the `magnitude` Score (None if not present)
    magnitude_ascending
        Whether to rank `magnitude` in ascending manner (None if not relevant)
    specificity
        Name of the `specificity` Score if Present (None if not present)
    specificity_ascending
        Whether to rank `magnitude` in ascending manner  (None if not relevant)
    permute
        Whether it requires permutations
    reference
        Publication reference in Harvard style

    Attributes
    ----------
    method_name
        Name of the Method
    complex_cols
        Columns relevant for protein complexes
    add_cols
        Additional columns required by the method
    fun
        Interaction Scoring function
    magnitude
        Name of the `magnitude` Score (None if not present)
    magnitude_ascending
        Whether to rank `magnitude` in ascending manner (None if not relevant)
    specificity
        Name of the `specificity` Score if Present (None if not present)
    specificity_ascending
        Whether to rank `magnitude` in ascending manner  (None if not relevant)
    permute
        Whether it requires permutations
    reference
        Publication reference in Harvard style
    instances
        List of instances of this class

    """

    # initiate a list to store weak references to all instances
    instances: list = []

    def __init__(self,
                 method_name: str,
                 complex_cols: list,
                 add_cols: list,
                 fun: Callable,
                 magnitude: str | None,
                 magnitude_ascending: bool | None,
                 specificity: str | None,
                 specificity_ascending: bool | None,
                 permute: bool,
                 reference: str
                 ):
        self.__class__.instances.append(weakref.proxy(self))
        self.method_name = method_name
        self.complex_cols = complex_cols
        self.add_cols = add_cols
        self.fun = fun
        self.magnitude = magnitude
        self.magnitude_ascending = magnitude_ascending
        self.specificity = specificity
        self.specificity_ascending = specificity_ascending
        self.permute = permute
        self.reference = reference  # type: ignore[assignment]

    def describe(self):
        """Briefly describes the method"""
        print(
            f"{self.method_name} uses `{self.magnitude}` and `{self.specificity}`"
            f" as measures of expression strength and interaction specificity, respectively"
        )

    def reference(self):
        """Prints out reference in Harvard format"""
        print(self.reference)

    def get_meta(self):
        """Returns method metadata as pandas row"""
        meta = DataFrame([{"Method Name": self.method_name,
                           "Magnitude Score": self.magnitude,
                           "Specificity Score": self.specificity,
                           "Reference": self.reference
                           }])
        return meta

    @d.dedent
    def by_sample(self,
                  adata: an.AnnData | MuData,
                  sample_key: str,
                  key_added: str = K.uns_key,
                  inplace: bool = V.inplace,
                  verbose: bool = V.verbose,
                  **kwargs) -> DataFrame | None:
        """
        Run a method by sample.

        Parameters
        ----------
        %(adata)s
        %(sample_key)s
        %(key_added)s
        %(inplace)s
        verbose
            Possible values: False, True, 'full', where 'full' will print the results for each sample,
            and True will only print the sample progress bar. Default is False.
        **kwargs
            Keyword arguments to pass to the method

        Raises
        ------
        ValueError
            If `sample_key` is not present in `adata.obs`

        Returns
        -------
        A pandas DataFrame with the results and a column sample is stored in `adata.uns[key_added]` if `inplace` is True,
        else the DataFrame is returned.

        """
        if sample_key not in adata.obs:
            raise ValueError(f"{sample_key} was not found in `adata.obs`.")

        if not adata.obs[sample_key].dtype.name == "category":
            _logg(f"Converting `{sample_key}` to categorical!", level='warn', verbose=verbose)
            adata.obs[sample_key] = adata.obs[sample_key].astype("category")

        if verbose == 'full':
            verbose = True
            full_verbose = True
        else:
            full_verbose = False

        samples = adata.obs[sample_key].cat.categories

        adata.uns[key_added] = {}

        progress_bar = tqdm(samples, disable=not verbose)
        for sample in (progress_bar):
            if verbose:
                progress_bar.set_description(f"Now running: {sample}")


            temp = adata[adata.obs[sample_key]==sample]
            if temp.isbacked:
                temp = temp.to_memory().copy() # NOTE does to_memory copy?
            else:
                temp = temp.copy()

            sample_res = self.__call__(temp, inplace=False, verbose=full_verbose, **kwargs)  # type: ignore[operator]

            adata.uns[key_added][sample] = sample_res

        liana_res = concat(adata.uns[key_added]).reset_index(level=1, drop=True).reset_index()
        liana_res = liana_res.rename({"index":sample_key}, axis=1)

        if inplace:
            adata.uns[key_added] = liana_res
        return None if inplace else liana_res


class Method(MethodMeta):
    """
    Ligand-Receptor Method Class

    Parameters
    ----------
    method
        Instance of metod metadata class

    Attributes
    ----------
    method
        Instance of metod metadata class

    """

    def __init__(self, _method: MethodMeta):
        super().__init__(method_name=_method.method_name,
                         complex_cols=_method.complex_cols,
                         add_cols=_method.add_cols,
                         fun=_method.fun,
                         magnitude=_method.magnitude,
                         magnitude_ascending=_method.magnitude_ascending,
                         specificity=_method.specificity,
                         specificity_ascending=_method.specificity_ascending,
                         permute=_method.permute,
                         reference=_method.reference  # type: ignore[arg-type]
                         )
        self._method = _method

    @d.dedent
    def __call__(self,
                 adata: an.AnnData | MuData,
                 groupby: str,
                 resource_name: str = V.resource_name,
                 expr_prop: float = V.expr_prop,
                 min_cells: int = V.min_cells,
                 groupby_pairs: DataFrame | None = V.groupby_pairs,
                 base: float = V.logbase,
                 supp_columns: list | None = V.supp_columns,
                 return_all_lrs: bool = V.return_all_lrs,
                 key_added: str = K.uns_key,
                 use_raw: bool | None = V.use_raw,
                 layer: str | None = V.layer,
                 de_method: str = V.de_method,
                 n_perms: int = V.n_perms,
                 seed: int = V.seed,
                 n_jobs: int = 1,
                 resource: DataFrame | None = V.resource,
                 interactions: list | None = V.interactions,
                 spatial_key: str = 'spatial',
                 spatial_kwargs: dict | None = None,
                 mdata_kwargs: dict | None = None,
                 inplace: bool = V.inplace,
                 verbose: bool | None = V.verbose,
                 ) -> DataFrame | None:
        """
        Run a ligand-receptor method.

        Parameters
        ----------
        %(adata)s
        %(groupby)s
        %(resource_name)s
        %(expr_prop)s
        %(min_cells)s
        %(groupby_pairs)s
        %(base)s
        supp_columns
            Additional columns to be added from any of the methods implemented in liana,
            or any of the columns returned by `scanpy.tl.rank_genes_groups`, each starting with ligand_* or receptor_*.
            For example, `['ligand_pvals', 'receptor_pvals']`. None by default.
        %(return_all_lrs)s
        %(key_added)s
        %(use_raw)s
        %(layer)s
        %(de_method)s
        %(n_perms)s
        %(seed)s
        n_jobs
            Number of jobs to run in parallel.
        %(resource)s
        %(interactions)s
        %(spatial_key)s
        %(spatial_kwargs)s
        %(mdata_kwargs)s
        %(inplace)s
        %(verbose)s

        Returns
        -------
        If ``inplace = False``, returns a `DataFrame` with ligand-receptor results
        Otherwise, modifies the ``adata`` object with the following key:
            - :attr:`anndata.AnnData.uns` ``[`key_added`]`` with the aforementioned DataFrame

        """
        if supp_columns is None:
            supp_columns = []
        if mdata_kwargs is None:
            mdata_kwargs = {}

        liana_res = liana_pipe(adata=adata,
                               groupby=groupby,
                               resource_name=resource_name,
                               resource=resource,
                               interactions=interactions,
                               expr_prop=expr_prop,
                               min_cells=min_cells,
                               supp_columns=supp_columns,
                               return_all_lrs=return_all_lrs,
                               groupby_pairs=groupby_pairs,
                               base=base,
                               de_method=de_method,
                               verbose=verbose,
                               _score=self._method,
                               n_perms=n_perms,
                               seed=seed,
                               n_jobs=n_jobs,
                               use_raw=use_raw,
                               layer=layer,
                               spatial_key=spatial_key,
                               spatial_kwargs=spatial_kwargs,
                               mdata_kwargs=mdata_kwargs
                               )
        if inplace:
            adata.uns[key_added] = liana_res
        return None if inplace else liana_res


def _show_methods(methods):
    return concat([method.get_meta() for method in methods])
