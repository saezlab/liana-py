import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, isspmatrix_csr


def zi_minmax(X: ArrayLike, cutoff: float = 0.5) -> csr_matrix:
    """
    Zero-inflated min-max scaling, adopted from CiteFuse (Kim et al., 2020; https://academic.oup.com/bioinformatics/article/36/14/4137/5827474).

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
    Each column is scaled independently, so the column minima land on 0 and the
    maxima on 1. The result is sparse; densified here to read it:

    >>> import numpy as np
    >>> from liana.utils import zi_minmax
    >>> x = np.array([[0.1, 0.3],
    ...               [2.0, 4.0],
    ...               [5.5, 7.1]])
    >>> zi_minmax(x).toarray().round(3)
    array([[0.   , 0.   ],
           [0.   , 0.544],
           [1.   , 1.   ]])

    `cutoff` is applied after scaling, so lowering it keeps more of the middle:

    >>> zi_minmax(x, cutoff=0.1).toarray().round(3)
    array([[0.   , 0.   ],
           [0.352, 0.544],
           [1.   , 1.   ]])

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
    >>> import numpy as np
    >>> from liana.utils import neg_to_zero
    >>> x = np.array([-1, -0.5, 0.1, 0.4, 2])
    >>> neg_to_zero(x).toarray()
    array([[0. , 0. , 0.1, 0.4, 2. ]])

    `cutoff` raises the threshold above 0:

    >>> neg_to_zero(x, cutoff=0.5).toarray()
    array([[0., 0., 0., 0., 2.]])

    """
    X = X.copy()
    if not isspmatrix_csr(X):
        X = csr_matrix(X)
    X.data[X.data < cutoff] = 0

    return X
