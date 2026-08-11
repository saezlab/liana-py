import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, isspmatrix_csr


def zi_minmax(X: ArrayLike, cutoff: float = 0.5) -> csr_matrix:
    """
    Zero-inflated min-max scaling, adopted from CiteFuse (Kim et al., 2020;
    https://academic.oup.com/bioinformatics/article/36/14/4137/5827474).

    This function scales the data to the range [0, 1] for each column of a
    two-dimensional array and sets values below a specified cutoff to 0 (after
    scaling).

    Parameters
    ----------
    X
        Data to be scaled.
    cutoff
        Cutoff value for zero-inflation - values less than this are set to 0.
        Default is 0.5.

    Returns
    -------
    X
        The scaled data matrix

    Examples
    --------
    >>> x = np.array([[0.1, 0.3],
    ...               [2.0, 4.0],
    ...               [5.5, 7.1]])
    >>> print(zi_minmax(x))
    <Compressed Sparse Row sparse matrix of dtype 'float64'
            with 6 stored elements and shape (3, 2)>
    Coords        Values
    (0, 0)        0.0
    (0, 1)        0.0
    (1, 0)        0.0
    (1, 1)        0.5441176470588236
    (2, 0)        1.0
    (2, 1)        1.0
    >>> print(zi_minmax(x, cutoff=0.1))
    <Compressed Sparse Row sparse matrix of dtype 'float64'
            with 6 stored elements and shape (3, 2)>
    Coords        Values
    (0, 0)        0.0
    (0, 1)        0.0
    (1, 0)        0.3518518518518518
    (1, 1)        0.5441176470588236
    (2, 0)        1.0
    (2, 1)        1.0

    """
    X = X.copy()
    if not isspmatrix_csr(X):
        X = csr_matrix(X)

    min_vals = np.array(X.min(axis=0).todense())[0]
    max_vals = np.array(X.max(axis=0).todense())[0]
    nonzero_rows, nonzero_cols = X.nonzero()
    scaled_values = (X.data - min_vals[nonzero_cols]) \
        / (max_vals[nonzero_cols] - min_vals[nonzero_cols])

    scaled_values[scaled_values < cutoff] = 0
    nonzero_rows, nonzero_cols = X.nonzero()

    X = csr_matrix(
        (scaled_values, (nonzero_rows, nonzero_cols)),
        shape=X.shape
    )

    return X


def neg_to_zero(X: ArrayLike, cutoff: float = 0) -> csr_matrix:
    """
    Set negative values to 0.

    Parameters
    ----------
    X
        Data to be transformed.
    cutoff
        Cutoff value for zero-inflation - values less than
        this are set to 0. Default is 0.

    Returns
    -------
    The modified data matrix

    Examples
    --------
    >>> x = np.array([-1, -0.5, 0.1, 0.4, 2])
    >>> print(neg_to_zero(x))
    <Compressed Sparse Row sparse matrix of dtype 'float64'
            with 5 stored elements and shape (1, 5)>
    Coords        Values
    (0, 0)        0.0
    (0, 1)        0.0
    (0, 2)        0.1
    (0, 3)        0.4
    (0, 4)        2.0
    >>> print(neg_to_zero(x, cutoff=0.5))
    <Compressed Sparse Row sparse matrix of dtype 'float64'
            with 5 stored elements and shape (1, 5)>
    Coords        Values
    (0, 0)        0.0
    (0, 1)        0.0
    (0, 2)        0.0
    (0, 3)        0.0
    (0, 4)        2.0
    """
    X = X.copy()
    if not isspmatrix_csr(X):
        X = csr_matrix(X)
    X.data[X.data < cutoff] = 0

    return X
