import numpy as np
from sklearn.neighbors import BallTree
from plotnine import ggplot, aes, geom_line, geom_point, theme_bw, xlab, ylab
from pandas import DataFrame

def query_bandwidth(coordinates, start=0, end=500, interval_n=50, reference=None):
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
         theme_bw(base_size=16) +
         xlab("Bandwidth") +
         ylab("Number of Neighbors")
         )
    
    return p, df
    