from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def sample_resource(adata: AnnData, n_lrs: int = 3000, seed: int = 1337) -> pd.DataFrame:
    resource = pd.DataFrame(product(adata.var_names, adata.var_names)).rename(columns={0: "ligand", 1: "receptor"})
    resource = resource[resource["ligand"] != resource["receptor"]]
    resource = resource.sample(n_lrs, replace=False, random_state=seed)
    return resource
