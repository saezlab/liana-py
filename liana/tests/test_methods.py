import pandas
from numpy import max, min
from numpy.testing import assert_almost_equal

from liana.method import cellphonedb, singlecellsignalr as sca, \
    natmi, connectome, logfc, geometric_mean, cellchat
    
from liana.testing._toy_adata import get_toy_adata

# load toy adata
adata = get_toy_adata()
expected_shape = adata.shape


def test_cellchat():
    cellchat(adata, groupby='bulk_labels', use_raw=True, n_perms=4)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'lr_probs' in liana_res.columns
    assert 'cellchat_pvals' in liana_res.columns
    assert max(liana_res[(liana_res.ligand == "TIMP1")].lr_probs) == 0.20561589810421071
    assert liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['lr_probs'].max() == 0.10198416583005679
    assert liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['cellchat_pvals'].mean() == 0.5125


def test_cellphonedb():
    cellphonedb(adata, groupby='bulk_labels', use_raw=True, n_perms=4)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'lr_means' in liana_res.columns
    assert 'cellphone_pvals' in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].lr_means), 2.134743630886078, decimal=6)
    assert_almost_equal(liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['lr_means'].max(), 1.4035000205039978, decimal=6)
    assert liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['cellphone_pvals'].mean() == 0.415


def test_cellphonedb_none():
    cellphonedb(adata, groupby='bulk_labels', use_raw=True, n_perms=None)
    assert adata.shape == expected_shape
    liana_res = adata.uns['liana_res']
    assert_almost_equal(liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['lr_means'].max(), 1.4035000205039978, decimal=6)
    assert 'cellphone_pvals' not in liana_res.columns 


def test_geometric_mean():
    geometric_mean(adata, groupby='bulk_labels', use_raw=True, n_perms=4)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'lr_gmeans' in liana_res.columns
    assert 'gmean_pvals' in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].lr_gmeans), 2.126363309240465, decimal=6)
    assert_almost_equal(liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['lr_gmeans'].max(), 1.4016519940029961, decimal=6)
    assert liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['gmean_pvals'].mean() == 0.5125


def test_natmi():
    natmi(adata, groupby='bulk_labels', use_raw=True)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'expr_prod' in liana_res.columns
    assert 'spec_weight' in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].spec_weight), 0.0604750001773605, decimal=6)
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].expr_prod), 4.521420922884062, decimal=6)
    assert_almost_equal(liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['spec_weight'].max(), 0.03480120361979308, decimal=6)
    assert_almost_equal(liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['expr_prod'].max(), 1.9646283122925752, decimal=6)


def test_sca():
    sca(adata, groupby='bulk_labels', use_raw=True)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'lrscore' in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].lrscore), 0.781133536891427, decimal=6)
    assert_almost_equal(liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['lrscore'].max(), 0.7017243729003677, decimal=6)


def test_logfc():
    logfc(adata, groupby='bulk_labels', use_raw=True)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'lr_logfc' in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].lr_logfc), 1.4352725744247437, decimal=6)
    assert_almost_equal(liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['lr_logfc'].max(), 1.0422011613845825, decimal=6)


def test_connectome():
    connectome(adata, groupby='bulk_labels', use_raw=True)
    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'expr_prod' in liana_res.columns
    assert 'scaled_weight' in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].scaled_weight), 0.9669451713562012, decimal=6)
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].expr_prod), 4.521420922884062, decimal=6)
    assert_almost_equal(liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['scaled_weight'].max(), 0.9002860486507416, decimal=6)
    assert_almost_equal(liana_res[liana_res['receptor_complex']=='CD74_CXCR4']['expr_prod'].max(), 1.9646283122925752, decimal=6)
    


def test_with_all_lrs():
    natmi(adata, groupby='bulk_labels', use_raw=True, return_all_lrs=True, key_added='all_res')
    lr_all = adata.uns['all_res']
    assert lr_all.shape == (4200, 15)
    assert all(lr_all[~lr_all.lrs_to_keep][natmi.magnitude] == min(lr_all[natmi.magnitude])) is True
    assert all(lr_all[~lr_all.lrs_to_keep][natmi.specificity] == min(lr_all[natmi.specificity])) is True


def test_methods_by_sample():
    logfc.by_sample(adata, groupby='bulk_labels', use_raw=True, return_all_lrs=True, sample_key='sample')
    lr_by_sample = adata.uns['liana_res']
    
    assert 'sample' in lr_by_sample.columns
    assert lr_by_sample.shape == (10836, 15)
    