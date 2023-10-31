import pandas as pd
import numpy as np

def _sample_interactions():
    columns = ['target', 'predictor', 'view', 'importances']
    interactions = pd.DataFrame(columns=columns)
    interactions['target'] = np.repeat(['a', 'b', 'c'], 3)
    interactions['predictor'] = np.tile(['x', 'y', 'z'], 3)
    interactions['view'] = np.repeat(['intra', 'inter', 'extra'], 3)
    interactions['importances'] = np.random.rand(9)

    return interactions


def _sample_target_metrics():
    columns = ['target', 'intra_R2', 'multi_R2', 'gain_R2', 'intra', 'extra']
    target_metrics = pd.DataFrame(columns=columns)
    target_metrics['target'] = ['a', 'b', 'c']
    target_metrics['intra_R2'] = np.random.rand(3)
    target_metrics['multi_R2'] = np.random.rand(3)
    target_metrics['gain_R2'] = np.random.rand(3)
    target_metrics['intra'] = np.random.rand(3)
    target_metrics['extra'] = np.random.rand(3)

    return target_metrics
