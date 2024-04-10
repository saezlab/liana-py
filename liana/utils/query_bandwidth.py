import numpy as np
from sklearn.neighbors import BallTree
from plotnine import ggplot, aes, geom_line, geom_point, theme_bw, xlab, ylab, scale_y_continuous
from pandas import DataFrame

def query_bandwidth(coordinates: np.ndarray,
                    start: int = 0,
                    end: int = 500,
                    interval_n:int = 50,
                    reference: np.ndarray = None
                    ):
    """
    Query the bandwidth (maximum distance) at which the average number of neighbors is maximized.

    Parameters
    ----------
    coordinates
        Spatial coordinates of spots.
    start
        Starting bandwidth.
    end
        Ending bandwidth.
    interval_n
        Number of bandwidths to query. Used to generate a linearly spaced interval between `start` and `end`.
        Default is 50.
    reference
        Reference coordinates to query the neighbors from. Default is `None`, which will use `coordinates`.

    Returns
    -------
    A `plotnine` plot and a `pandas` DataFrame with the following columns:
        - `bandwith`: the bandwidth (maximum distance) at which the average number of neighbors is maximized.
        - `neighbours`: the average number of neighbors at the specified bandwidth.
    """
    tree = BallTree(coordinates, metric='euclidean')
    df = DataFrame()
    interval = np.linspace(start, end, interval_n)

    if reference is None:
        _reference = coordinates
    else:
        _reference = reference

    for n in range(interval_n):
        max_distance = interval[n]
        df.loc[n, 'bandwith'] = max_distance

        # query the neighbors within the specified distance
        num_neighbors = tree.query_radius(_reference, r=max_distance, count_only=True)

        # calculate the average number of neighbors
        avg_nn = np.mean(num_neighbors)
        df.loc[n, 'neighbours'] = avg_nn

    p = (ggplot(df, aes(x='bandwith', y='neighbours')) +
         geom_line() +
         geom_point() +
         scale_y_continuous(breaks=range(start, end, interval_n)) +
         theme_bw(base_size=16) +
         xlab("Bandwidth") +
         ylab("Number of Neighbors")
         )

    return p, df
