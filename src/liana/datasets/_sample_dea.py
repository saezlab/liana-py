from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from liana._core._types import get_obs

if TYPE_CHECKING:
    from anndata import AnnData


def _sample_dea(adata: AnnData, groupby: str) -> pd.DataFrame:
    nrow = adata.n_vars * 5

    rng = np.random.default_rng(1337)

    dea_df = pd.DataFrame(
        {
            "gene": rng.choice(adata.var_names, nrow),
            "stat": rng.random(nrow),
            "pval": rng.random(nrow),
            "padjusted": rng.random(nrow),
            groupby: rng.choice(get_obs(adata)[groupby].unique(), nrow),
        }
    )
    dea_df = dea_df.drop_duplicates(["gene", groupby]).set_index("gene")

    return dea_df
