import numpy as np
import pandas as pd
from liana.method import estimate_metalinks
from liana.testing._sample_anndata import generate_toy_adata

def test_estimate_metalinks():
    adata = generate_toy_adata()
    metabolites = ['A', 'A','B','B']
    resource = pd.DataFrame({'metabolite': np.unique(metabolites),
                             'receptor': ['TNFRSF4', 'ITGB2']})
    pd_net = pd.DataFrame({'metabolite': metabolites,
                          'target': ['RBP7', 'SRM', 'MAD2L2', 'AGTRAP'],
                          'weight': [0.1, 0.2, 0.3, 1]})
    t_net = pd.DataFrame({'metabolite': metabolites,
                         'target': ['HES4', 'TNFRSF4', 'SSU72', 'PARK7'],
                         'weight': [0.1, 0.2, 0.3, 1]})

    mdata = estimate_metalinks(adata, resource,
                               pd_net, t_net,
                               source='metabolite',
                               use_raw=True, verbose=True,
                               min_n=2)
    assert np.isin(['metabolite', 'receptor'], list(mdata.mod.keys())).all()
    assert (mdata.var.index == ['A', 'B', 'ITGB2', 'TNFRSF4']).all()
    np.testing.assert_almost_equal(mdata.mod['metabolite'].X.mean(), -0.18889697, decimal=5)
    np.testing.assert_almost_equal(mdata.mod['receptor'].X.mean(), 0.7754228, decimal=5)
