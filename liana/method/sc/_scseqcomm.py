from liana.method.sc._Method import Method, MethodMeta
import numpy as np

def _inter_score(x):
    inter_score = np.minimum(x['ligand_cdf'], x['receptor_cdf'])
    return inter_score, None

_scseqcomm = MethodMeta(method_name="scSeqComm",
                        complex_cols=["ligand_means", "receptor_means"],
                        add_cols=["ligand_cdf", "receptor_cdf"],
                        fun=_inter_score,
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
