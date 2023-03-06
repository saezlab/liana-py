from liana.method.ml._ml_Method import MetabMethod, MetabMethodMeta
from liana.method.ml.scores import _product_score, _simple_prod
#from liana.method.ml.estimations import mean_per_cell





# Initialize metalinks Meta
_metalinks = MetabMethodMeta(score_method_name="product_score",
                            complex_cols=['ligand_means', 'receptor_means'],
                            add_cols=['ligand_means_sums', 'receptor_means_sums'],
                            fun=_product_score,
                            magnitude='metalinks_score',
                            magnitude_ascending=False, 
                            specificity='pval',
                            specificity_ascending=True,  
                            permute=True,
                            agg_fun=_simple_prod, #check again
                            score_reference='Saez-Lab',
                            est_reference=None
                            
                    )

# Initialize callable Method instance
metalinks = MetabMethod(_SCORE=_metalinks)




