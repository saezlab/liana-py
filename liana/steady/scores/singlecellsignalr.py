import numpy as np

from liana.steady.Method import Method, MethodMeta


# Internal Function to calculate SingleCellR LRscore
def _sca_score(x):
    lr_sqrt = np.sqrt(x.ligand_means) * np.sqrt(x.receptor_means)
    return lr_sqrt / (lr_sqrt + x.mat_mean), None


# Initialize CPDB Meta
_singlecellsignalr = MethodMeta(method_name="SingleCellSignalR",
                                complex_cols=['ligand_means', 'receptor_means'],
                                add_cols=['mat_mean'],
                                fun=_sca_score,
                                magnitude='lrscore',
                                magnitude_desc=True,
                                specificity=None,
                                specificity_desc=None,
                                permute=False,
                                reference=''
                                )

# Initialize callable Method instance
singlecellsignalr = Method(_SCORE=_singlecellsignalr)
