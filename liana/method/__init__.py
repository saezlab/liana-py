from liana.method.sc._Method import Method, MethodMeta, _show_methods
from liana.method.sc._rank_aggregate import AggregateClass, _rank_aggregate_meta as aggregate_meta
from liana.method.sc import cellphonedb, connectome, logfc, natmi, singlecellsignalr, geometric_mean, cellchat, scseqcomm

from liana.method.sp import bivar, lr_bivar, genericMistyData, lrMistyData, MistyData
from liana.method.fun._causalnet import find_causalnet, build_prior_network

import numpy as np

# callable consensus instance
_methods = [cellphonedb, connectome, logfc, natmi, singlecellsignalr, cellchat]
rank_aggregate = AggregateClass(aggregate_meta, methods=_methods)


def show_methods():
    """Shows methods available in LIANA"""
    return _show_methods(_methods + [rank_aggregate, geometric_mean])

def get_method_scores():
    """Returns a dict of all scoring functions, with a boolean indicating whether the score is ascending or not"""
    instances = np.array(MethodMeta.instances)
    relevant = np.array([(isinstance(instance, Method)) | (isinstance(instance, AggregateClass)) for instance in instances])
    instances = instances[relevant]

    specificity_scores = {method.specificity: method.specificity_ascending for method in instances if method.specificity is not None}
    magnitude_scores = {method.magnitude : method.magnitude_ascending for method in instances if method.magnitude is not None}

    scores = {**specificity_scores, **magnitude_scores}
    return scores
