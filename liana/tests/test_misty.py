import os
import pathlib
import numpy as np
import scanpy as sc
from mudata import MuData
from liana.method.sp._misty import misty
from liana.method.sp._misty_constructs import lrMistyData, genericMistyData

test_path = pathlib.Path(__file__).parent

adata = sc.read_h5ad(os.path.join(test_path, "data" , "synthetic.h5ad"))
adata = sc.pp.subsample(adata, n_obs=100, copy=True)
mdata = MuData({'rna':adata})


### TODO: TEST SHAPES BEFORE CHANGING TO MISTYDATA

def test_misty_para():
    misty(mdata=mdata, x_mod="rna",
          bandwidth=10,
          alphas=[0.1, 1, 10],
          add_juxta=False,
          set_diag=False,
          seed=42
          )
    
    # misty = genericMistyData(adata, bandwidth=10, add_juxta=False, set_diag=False, seed=42)
    # misty(bypass_intra=False, seed=42)
    # NOTE: importances are not always 1 per target per view? -> why?
    # list(importances.groupby(['target', 'view']).sum()['value'].values)
    
    misty_res = mdata.uns['misty_results']
    assert np.isin(list(misty_res.keys()), ['target_metrics', 'importances']).all()
    # assert that we get contribution = 1 for each target (11 targets in total)
    assert np.sum(misty_res['target_metrics'][['intra', 'para']].values, axis=1).sum() == 11.0
    # assert that we get importances = 1 per target per view
    target = 'ECM' # 1 target, and we have 2 views
    importances = misty_res['importances']
    assert importances[importances['target']==target]['value'].sum() == 2.0
    interaction_msk = (importances['target']=='ligA') & \
        (importances['predictor']=='protE')
    np.testing.assert_almost_equal(importances[interaction_msk]['value'].values,
                                   np.array([0.00162521, 0.05350179]))
    # assert that R2 gain is consistent
    assert misty_res['target_metrics']['gain.R2'].mean() == -0.003487056749054847
    

def test_misty_bypass():
    misty(mdata=mdata, x_mod="rna",
          bandwidth=10, alphas=1,
          bypass_intra=True, add_juxta=True, 
          set_diag=True, seed=42,
          overwrite=True)
    
    # misty = genericMistyData(adata, bandwidth=10, add_juxta=True, set_diag=True, cutoff=0, coord_type="generic", delaunay=True)
    # misty(alphas=1, bypass_intra=True, seed=42)
    
    
    misty_res = mdata.uns['misty_results']
    # multi & gain should be identical here (gain.R2 = multi.R2 - 0; when intra is bypassed)
    assert misty_res['target_metrics']['gain.R2'].equals(misty_res['target_metrics']['multi.R2'])
    assert misty_res['target_metrics']['multi.R2'].sum() == -2.171910172942944
    # ensure both para and juxta are present in contributions
    assert np.isin(['juxta', 'para'], misty_res['target_metrics'].columns).all()

def test_misty_groups():    
    misty(mdata=mdata,
          x_mod="rna",
          bandwidth=20,
          seed=42,
          alphas=1,
          set_diag=True,
          keep_same_predictor=True, # TODO: Rename these two
          group_env_by='cell_type',
          group_intra_by='cell_type',
          bypass_intra=False, # TODO: shouldn't this always be false when keep=True?
          overwrite=True
          )
    
    # misty = genericMistyData(adata, bandwidth=20, add_juxta=True, set_diag=True, cutoff=0, coord_type="generic", delaunay=True)
    # misty(alphas=1, bypass_intra=False, seed=42, keep_same_predictor=True, group_env_by='cell_type', group_intra_by='cell_type')
    
    misty_res = mdata.uns['misty_results'].copy()
    importances = misty_res['importances']
    top5 = importances.sort_values('value', ascending=False)['target'][0:5].values
    assert (top5 == np.array(['prodB', 'prodB', 'prodC', 'prodC', 'ligB'])).all()
    
    # assert that there are self interactions = var_n * var_n
    self_interacctions = importances[(importances['target']==importances['predictor'])]
    # assert all intra is false - cannot predict itself
    assert self_interacctions[self_interacctions['view']=='intra']['value'].isna().all()
    # 11 vars * 4 envs * 3 views = 132
    assert self_interacctions.shape == (132, 6)
    
    perf_actual = (misty_res['target_metrics'].
     groupby(['intra_group', 'env_group'])['gain.R2'].
     mean().values
    )
    perf_expected = np.array([0.07707842154368322, 0.03275119745812092, 0.12220937386420569, 0.05019178543805167])
    np.testing.assert_almost_equal(perf_actual, perf_expected)
