from __future__ import annotations

import numpy as np
import pandas as pd
import plotnine as p9
from anndata import AnnData
from sklearn.decomposition import NMF
from tqdm import tqdm

from liana._docs import d
from liana._logging import _check_if_installed, _logg
from liana.method._pipe_utils._pre import _choose_mtx_rep


@d.dedent
def nmf(adata: AnnData = None,
        df: pd.DataFrame = None,
        n_components: int | None = None,
        k_range: range = range(1, 11),
        use_raw: bool = False,
        layer: str | None = None,
        inplace: bool = True,
        verbose: bool = False,
        **kwargs
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None, int] | None:
    """
    Fits NMF to an AnnData object.

    Parameters
    ----------
    %(adata)s
    df
        Alternative input for data as a `DataFrame`, only used if `adata` is None.
    n_components
        Number of components to use. If None, the number of components is estimated using the elbow method.
    k_range
        Range of components to test. Default: range(1, 10).
    %(use_raw)s
    %(layer)s
    %(inplace)s
    **kwargs
        Keyword arguments to pass to ``sklearn.decomposition.NMF``.

    Returns
    -------
    If inplace is True, it will add ``NMF_W`` and ``NMF_H`` to the ``adata.obsm`` and ``adata.varm``.
    If n_components is None, it will also add ``nfm_errors`` and ``nfm_rank`` to ``adata.uns``.

    If inplace is False, it will return ``W`` and ``H``, and if n_components is None, it will also return ``errors`` and ``n_components``.
    If n_components is None and inplace, ``errors`` and ``n_components`` will be assigned to ``adata.uns``.
    If ``df`` is provided, inplace is always False.

    Raises
    ------
        ValueError
            If `adata` is provided but it's not a valid instance of an `AnnData` object or neither an `AnnData` or `DataFrame` intance is provided as input

    Examples
    --------
    ``nmf`` expects a *non-negative* matrix -- typically the local ligand-receptor
    scores from ``liana.method.bivariate``:

    >>> import liana as li
    >>> adata = li.testing.generate_toy_spatial()
    >>> lrdata = li.mt.bivariate(adata, resource_name='consensus',
    ...                          local_name='cosine', global_name=None,
    ...                          n_perms=None)
    >>> li.multi.nmf(lrdata, n_components=3, random_state=0)

    Leaving `n_components` as `None` instead estimates the rank with
    :func:`liana.multi.estimate_elbow` and draws the elbow plot.

    Read the factors out with :func:`liana.utils.get_factor_scores` and
    :func:`liana.utils.get_variable_loadings`.

    """
    if adata is not None:
        if isinstance(adata, AnnData):
            X = _choose_mtx_rep(adata, layer=layer, use_raw=use_raw)
        else :
            raise ValueError('Provide an AnnData object.')
    elif df is not None:
        X = df.values
    else:
        raise ValueError('Provide either an AnnData object or a DataFrame.')

    if n_components is None:
        errors, n_components = estimate_elbow(X, k_range=k_range, verbose=verbose, **kwargs)
        _plot_elbow(errors, n_components)
    else:
        errors, n_components = None, n_components

    nmf = NMF(n_components=n_components, **kwargs)
    W = nmf.fit_transform(X)
    H = nmf.components_.T

    inplace = inplace and (adata is not None)
    if inplace:
        adata.obsm['NMF_W'] = W
        adata.varm['NMF_H'] = H
        adata.uns['nmf_errors'] = errors
        adata.uns['nmf_rank'] = n_components

    return None if inplace else (W, H, errors, n_components)


def estimate_elbow(X, k_range, verbose=False, **kwargs):
    """
    Estimate the rank of an NMF factorization from the elbow of its error curve.

    Parameters
    ----------
    X
        Non-negative matrix to factorize.
    k_range
        Ranks to fit. The elbow is located among these, so `None` is returned if
        no knee is found within them.
    verbose
        Whether to show a progress bar and report the estimated rank.
    kwargs
        Keyword arguments passed to :class:`sklearn.decomposition.NMF`.

    Returns
    -------
    A tuple of the reconstruction error per rank (a `DataFrame` with columns
    `k` and `error`) and the estimated rank.

    Examples
    --------
    Called by :func:`liana.multi.nmf` when `n_components` is `None`. Unlike `nmf`
    it takes a plain non-negative matrix, not an AnnData. This one is built from
    two blocks, so its true rank is 2:

    >>> import numpy as np
    >>> import liana as li
    >>> W = np.repeat(np.eye(2), 6, axis=0)
    >>> H = np.array([[3., 2., 1., 0., 0., 0.],
    ...               [0., 0., 0., 1., 2., 3.]])
    >>> errors, rank = li.multi.estimate_elbow(W @ H, k_range=range(1, 6),
    ...                                        random_state=0, max_iter=500)

    `rank` is the knee of the error curve -- 2 here, since the error collapses as soon
    as `k` reaches the true rank and cannot improve after.

    If no knee can be located within `k_range`, `rank` comes back as `None` --
    widen the range. A `k_range` that starts above the true rank returns its own
    lowest value.

    """
    kn = _check_if_installed('kneed')
    errors = []
    for k in tqdm(k_range, disable=not verbose):
        error = _calculate_error(X, k, **kwargs)
        errors.append(error)

    kneedle = kn.KneeLocator(x=k_range,
                             y=errors,
                             direction='decreasing',
                             curve='convex',
                             interp_method='interp1d',
                             S=1
                             )
    rank = kneedle.knee

    _logg(f'Estimated rank: {rank}', verbose=verbose)

    errors = pd.DataFrame(errors,
                          index=list(k_range),
                          columns=['error']). \
                              reset_index().rename(columns={'index': 'k'})

    return errors, rank


def _calculate_error(X, n_components, **kwargs):
    nmf = NMF(n_components=n_components, **kwargs)
    W = nmf.fit_transform(X)
    H = nmf.components_

    Xhat = np.dot(W, H)
    error = np.mean(np.abs(X - Xhat))

    return error


def _plot_elbow(errors, n_components, x='k', y='error'):
    p = (
        p9.ggplot(errors, p9.aes(x=x, y=y)) +
        p9.geom_line() +
        p9.geom_point() +
        p9.theme_bw() +
        p9.scale_x_continuous(breaks=errors[x].values) +
        p9.labs(x='Component number (k)', y='Reconstruction error') +
        p9.geom_vline(xintercept=n_components, linetype='dashed', color='red')
    )
    p.draw()
