
from pandas import Series
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr, hstack
from anndata import AnnData

def _add_complexes_to_var(adata, entities, complex_sep='_'):
    """
    Generate an AnnData object with complexes appended as variables.
    """
    complexes = entities[Series(entities).str.contains(complex_sep)]
    X = None
    for comp in complexes:
        subunits = comp.split(complex_sep)

        # keep only complexes, the subunits of which are in var
        if all([subunit in adata.var.index for subunit in subunits]):
            adata.var.loc[comp, :] = None

            # create matrix for this complex
            new_array = csr_matrix(adata[:, subunits].X.min(axis=1))

            if X is None:
                X = new_array
            else:
                X = hstack((X, new_array))

    adata = AnnData(X=hstack((adata.X, X)),
                    obs=adata.obs,
                    var=adata.var,
                    obsm=adata.obsm,
                    varm = adata.varm,
                    obsp=adata.obsp,
                    uns = adata.uns,
                    )

    if not isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()

    return adata


def _zscore(mat, axis=0, global_r=False):
    if global_r: # NOTE: specific to global SpatialDM
        spot_n = 1
    else:
        spot_n = mat.shape[axis]

    mat = mat - mat.mean(axis=axis)
    mat = mat / np.sqrt(np.sum(np.power(mat, 2), axis=axis) / spot_n)
    mat = np.clip(mat, -10, 10)

    return np.array(mat)

def _spatialdm_weight_norm(weight):
    norm_factor = weight.shape[0] / weight.sum()
    weight = norm_factor * weight
    return weight
