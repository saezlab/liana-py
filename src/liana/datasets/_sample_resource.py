from itertools import product

import pandas as pd


def sample_resource(adata, n_lrs = 3000, seed=1337):
    resource = pd.DataFrame(product(adata.var_names, adata.var_names)).rename(columns={0: "ligand", 1: "receptor"})
    resource = resource[resource["ligand"] != resource["receptor"]]
    resource = resource.sample(n_lrs, replace=False, random_state=seed)
    return resource
