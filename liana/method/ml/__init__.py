# from ._ml_Method import _show_met_est_methods
# from .met_inf._mrank_aggregate import MetabAggregateClass, _mrank_aggregate_meta
# from .scores import mebocost


# # callable consensus instance
# _mmethods = [mebocost]
# rank_aggregate = MetabAggregateClass(_mrank_aggregate_meta, methods=_mmethods)


# def show_met_est_methods():
#      """Shows metabolic estimation methods available in LIANA"""
#      return _show_met_est_methods(_mmethods + [rank_aggregate])



# def get_method_scores():
#     """Returns a dict of all scoring functions, with a boolean indicating whether the score is ascending or not"""
#     instances = np.array(MethodMeta.instances)
#     relevant = np.array([(isinstance(instance, Method)) | (isinstance(instance, AggregateClass)) for instance in instances])
#     instances = instances[relevant]
#     specificity_scores = {method.specificity: method.specificity_ascending for method in instances if method.specificity is not None}
#     magnitude_scores = {method.magnitude : method.magnitude_ascending for method in instances if method.magnitude is not None}
#     scores = {**specificity_scores, **magnitude_scores}
#     return scores
