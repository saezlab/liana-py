import numpy as np

from ..method import get_method_scores

def _process_scores(liana_res, score_key, inverse_fun):

    df = liana_res.copy()
    scores = get_method_scores()

    if not np.isin(score_key, list(scores.keys())).any():
        raise ValueError(f"Score column {score_key} not found in liana's method scores. ")

    # reverse if ascending order
    ascending_order = scores[score_key]
    if(ascending_order):
        df[score_key] = inverse_fun(df[score_key])

    return df
