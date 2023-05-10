import os
import pathlib
import numpy as np
import scanpy as sc
import muon as mu
from liana.method.sp._misty import misty

test_path = pathlib.Path(__file__).parent

adata = sc.read_h5ad(os.path.join(test_path, "data" , "synthetic.h5ad"))
mdata = mu.MuData({'rna':adata})


def test_misty_para():
    misty(mdata=mdata, x_mod="rna", bandwidth=10, add_juxta=False, set_diag=False, seed=42)
    
    misty_res = mdata.uns['misty_results']
    assert np.isin(list(misty_res.keys()), ['performances', 'contributions', 'importances']).all()
    # assert that we get contribution = 1 for each target (11 targets in total)
    assert np.sum(misty_res['contributions'][['intra', 'para']].values, axis=1).sum() == 11.0
    # assert that we get importances = 1 per target per view
    target = 'ECM' # 1 target, and we have 2 views
    importances = misty_res['importances']
    assert importances[importances['target']==target]['value'].sum() == 2.0
    interaction_msk = (importances['target']=='ligA') & \
        (importances['predictor']=='protE')
    np.testing.assert_almost_equal(importances[interaction_msk]['value'].values,
                                   np.array([0.00044188, 0.09210615]))
    # assert that R2 gain is consistent
    assert misty_res['performances']['gain.R2'].mean() == 0.007905217610502951
    

def test_misty_bypass():
    misty(mdata=mdata, x_mod="rna", bandwidth=10, bypass_intra=True, add_juxta=True, set_diag=True, seed=42, overwrite=True)
    misty_res = mdata.uns['misty_results']
    # multi & gain should be identical here (gain.R2 = multi.R2 - 0; when intra is bypassed)
    assert misty_res['performances']['gain.R2'].equals(misty_res['performances']['multi.R2'])
    assert misty_res['performances']['multi.R2'].sum() == 3.1041304499677613
    # ensure both para and juxta are present in contributions
    assert np.isin(['juxta', 'para'], misty_res['contributions'].columns).all()

def test_misty_groups():
    
    misty(mdata=mdata, x_mod="rna", bandwidth=20, seed=42,
          set_diag=True, keep_same_predictor=True, # TODO: Rename these two
          group_env_by='cell_type', group_intra_by='cell_type',
          bypass_intra=False, # TODO: shouldn't this always be false when keep=True?
          overwrite=True
          )
    misty_res = mdata.uns['misty_results'].copy()
    importances = misty_res['importances']
    
    # assert that there are self interactions = var_n * var_n
    self_interacctions = importances[(importances['target']==importances['predictor'])]
    # assert all intra is false - cannot predict itself
    assert self_interacctions[self_interacctions['view']=='intra']['value'].isna().all()
    # 11 vars * 4 envs * 3 views = 132
    assert self_interacctions.shape == (132, 6)
    
    perf_actual = (misty_res['performances'].
     groupby(['intra_group', 'env_group'])['gain.R2'].
     mean().values
    )
    perf_expected = np.array([0.01033989, 0.00859758, 0.00973386, 0.01135016])
    np.testing.assert_almost_equal(perf_actual, perf_expected)
