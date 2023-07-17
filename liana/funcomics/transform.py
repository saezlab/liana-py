from scipy.sparse import csr_matrix, isspmatrix_csr
import numpy as np

def zi_minmax(X, cutoff):    
    X = X.copy()
    # Ensure the matrix is in Compressed Sparse Row (CSR) format
    if not isspmatrix_csr(X):
        X = csr_matrix(X)

    # Min-Max scaling on non-zero elements
    min_vals = np.array(X.min(axis=0).todense())[0]
    max_vals = np.array(X.max(axis=0).todense())[0]
    nonzero_rows, nonzero_cols = X.nonzero()
    scaled_values = (X.data - min_vals[nonzero_cols]) / (max_vals[nonzero_cols] - min_vals[nonzero_cols])
    
    # Apply cutoff to the matrix
    scaled_values[scaled_values < cutoff] = 0
    nonzero_rows, nonzero_cols = X.nonzero()

    # Create a new sparse matrix with scaled values
    X = csr_matrix((scaled_values, (nonzero_rows, nonzero_cols)), shape=X.shape)

    return X
