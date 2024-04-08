import numpy as np

from liana.method.sc._Method import Method, MethodMeta, _show_methods
from liana.method.sc._rank_aggregate import AggregateClass, _rank_aggregate_meta as aggregate_meta
from liana.method.sc import cellphonedb, connectome, logfc, natmi, singlecellsignalr, geometric_mean, cellchat, scseqcomm

from liana.method.sp import bivariate, genericMistyData, lrMistyData, MistyData
from liana.method.fun._causalnet import find_causalnet, build_prior_network
from liana._constants import DefaultValues as V

# callable consensus instance
_methods = [cellphonedb, connectome, logfc, natmi, singlecellsignalr]
rank_aggregate = AggregateClass(aggregate_meta, methods=_methods)


def show_methods():
    """Shows methods available in LIANA"""
    return _show_methods(_methods + [rank_aggregate, geometric_mean, scseqcomm, cellchat])

def get_method_scores():
    """Returns a dict of all scoring functions, with a boolean indicating whether the score is ascending or not"""
    instances = np.array(MethodMeta.instances)
    relevant = np.array([(isinstance(instance, Method)) | (isinstance(instance, AggregateClass)) for instance in instances])
    instances = instances[relevant]

    specificity_scores = {method.specificity: method.specificity_ascending for method in instances if method.specificity is not None}
    magnitude_scores = {method.magnitude : method.magnitude_ascending for method in instances if method.magnitude is not None}

    scores = {**specificity_scores, **magnitude_scores}
    return scores

def process_scores(liana_res, score_key, inverse_fun=V.inverse_fun):

    df = liana_res.copy()
    scores = get_method_scores()

    if not np.isin(score_key, list(scores.keys())).any():
        raise ValueError(f"Score column {score_key} not found in liana's method scores.")

    # reverse if ascending order
    ascending_order = scores[score_key]
    if(ascending_order):
        df[score_key] = inverse_fun(df[score_key])

    return df
