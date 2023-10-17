import liana as li
import scanpy as sc
import numpy as np
import pandas as pd
from liana.method import scseqcomm
from liana.testing._sample_anndata import generate_toy_adata
from numpy.testing import assert_almost_equal

"""
#### TEST WITH PBMC DATA ###
adata = sc.datasets.pbmc68k_reduced()
print(adata.var.loc['S100A9'])
print(type(adata.raw.X))
#print(adata.X)
input("press...")
#adata.write_h5ad(filename='pbmc3k_processed.h5ad')

scseqcomm(adata, groupby='bulk_labels', use_raw=True, expr_prop=0, return_all_lrs = True, key_added = 'scseqcomm_res')

print(adata.uns['scseqcomm_res'].shape)
print(adata.uns['scseqcomm_res'])
print(adata.uns['scseqcomm_res'].columns)
res = pd.DataFrame()
res['interaction'] = adata.uns['scseqcomm_res']['ligand_complex'] + '&' + \
            adata.uns['scseqcomm_res']['receptor_complex'] + '&' + \
            adata.uns['scseqcomm_res']['source'] + '&' + \
            adata.uns['scseqcomm_res']['target']
res['score'] = adata.uns['scseqcomm_res']['inter_score']
res['ligand_score'] = adata.uns['scseqcomm_res']['ligand_score']
res['receptor_score'] = adata.uns['scseqcomm_res']['receptor_score']
res.to_csv('interaction_post_scoring.csv', index=False)
"""

#### TEST WITH LIANA GENERATE TOY DATA ###
adata = generate_toy_adata()
print(adata)

# THIS DATA IS ACTUALLY THE SAME AS THE DATA FROM PBMC68K_reduced
expected_shape = adata.shape
print("expected_shape: {}".format(expected_shape))

scseqcomm(adata,groupby='bulk_labels', use_raw=True, expr_prop=0, return_all_lrs=True)

assert adata.shape == expected_shape
assert 'liana_res' in adata.uns.keys()

liana_res = adata.uns['liana_res']
assert isinstance(liana_res, pd.DataFrame)

assert 'inter_score' in liana_res.columns

assert_almost_equal(liana_res[(liana_res.ligand == "TIMP1") & \
                            (liana_res.receptor == "CD63") & \
                            (liana_res.source == "Dendritic") & \
                            (liana_res.target == "CD4+/CD45RA+/CD25- Naive T")].inter_score.values, 0.6819619345, decimal = 5)
assert_almost_equal(max(liana_res[(liana_res.receptor_complex == "CD74_CXCR4")].inter_score), 0.9997214654, decimal = 6)

