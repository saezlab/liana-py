import numpy as np
import pandas as pd


def _sample_dea(adata, groupby):
    nrow = adata.n_vars * 5

    rng = np.random.default_rng(1337)

    dea_df = pd.DataFrame({'gene': rng.choice(adata.var_names, nrow),
                            'stat': rng.random(nrow),
                            'pval': rng.random(nrow),
                            'padjusted': rng.random(nrow),
                            groupby: rng.choice(adata.obs[groupby].unique(), nrow)
                            })
    dea_df = dea_df.drop_duplicates(['gene', groupby]).set_index("gene")

    return dea_df
