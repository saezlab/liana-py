import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm

from anndata import AnnData
from pandas import DataFrame
from typing import Optional

from tqdm import tqdm

from liana.method._global_lr_pipe import _global_lr_pipe
from liana.method.sp._spatial_utils import _local_to_dataframe, _local_permutation_pvals



# TODO MethodMeta class to generalizable, SpatialMethod & SingleCellMethod should inherit and extend
class SpatialMethod:
    """
    A SpatialMethod Class
    """

    def __init__(self,
                 method_name: str,
                 key_cols: list,  # note that this is defined here but not in Method
                 reference: str
                 ):
        """
        Parameters
        ----------
        method_name
            Name of the Method
        key_cols
            columns which make every interaction unique (i.e. PK).
        reference
            Publication reference in Harvard style
        """
        self.method_name = method_name
        self.key_cols = key_cols
        self.reference = reference

    def describe(self):
        """Briefly described the method"""
        print(f"{ self.method_name } does XYZ")

    def reference(self):
        """Prints out reference in Harvard format"""
        print(self.reference)

    def get_meta(self):
        """Returns method metadata as pandas row"""
        meta = DataFrame([{"Method Name": self.method_name,
                           "Reference": self.reference
                           }])
        return meta




class SpatialDM(SpatialMethod):
    def __init__(self, _method, _obsm_keys):
        super().__init__(method_name=_method.method_name,
                         key_cols=_method.key_cols,
                         reference=_method.reference,
                         )

        self.obsm_keys = _obsm_keys
        self._method = _method

    def __call__(self,
                 adata: AnnData, ## TODO mats or adatas?
                 expr_prop: float = 0.05,
                 n_perm: int = 100,
                 positive_only: bool = True, ## TODO - both directions?
                 use_raw: Optional[bool] = True,
                 layer: Optional[str] = None,
                 verbose: Optional[bool] = False,
                 seed: int = 1337,
                 resource: Optional[DataFrame] = None,
                 inplace=True
                 ):

        XXXX
    
    
    
    
    
# initialize instance
_spatialmethod = SpatialMethod(
    method_name="Bivariate Coexpressions",
    key_cols=[],
    reference=""
)

spatial_method = SpatialDM(_method=_spatialmethod,
                           _obsm_keys=['proximity']
                      )
