from liana.method.sc._Method import Method, MethodMeta
import numpy as np


def _lr_score(x, axis=0):
    inter_score = np.minimum(x[0],x[1])
    return inter_score
    
def _intercellular_score(x):
    inter_score = _lr_score((x['ligand_score'], x['receptor_score']))

    return inter_score, None

_scseqcomm = MethodMeta(method_name="scSeqComm",
                        complex_cols=["ligand_means","receptor_means"],
                        add_cols=["ligand_score","receptor_score"],
                        fun=_intercellular_score,
                        magnitude="inter_score",
                        magnitude_ascending=False,
                        specificity=None,
                        specificity_ascending=None,
                        permute=False,
                        reference="Baruzzo, G., Cesaro, G., Di Camillo, B. "
                                  "2022. Identify, quantify and characterize cellular communication "
                                  "from single-cell RNA-sequencing data with scSeqComm. Bioinformatics, "
                                  "38(7), pp.1920-1929"
                        )
scseqcomm = Method(_method=_scseqcomm)