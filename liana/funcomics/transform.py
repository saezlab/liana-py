from scipy.sparse import csr_matrix, isspmatrix_csr
import numpy as np

def zi_minmax(X, cutoff):    
    X = X.copy()
    if not isspmatrix_csr(X):
        X = csr_matrix(X)

    min_vals = np.array(X.min(axis=0).todense())[0]
    max_vals = np.array(X.max(axis=0).todense())[0]
    nonzero_rows, nonzero_cols = X.nonzero()
    scaled_values = (X.data - min_vals[nonzero_cols]) / (max_vals[nonzero_cols] - min_vals[nonzero_cols])
    
    scaled_values[scaled_values < cutoff] = 0
    nonzero_rows, nonzero_cols = X.nonzero()

    X = csr_matrix((scaled_values, (nonzero_rows, nonzero_cols)), shape=X.shape)

    return X

def neg_to_zero(X):
    X = X.copy()
    if not isspmatrix_csr(X):
        matrix = csr_matrix(X)
    X.data[X.data < 0] = 0
    return X
