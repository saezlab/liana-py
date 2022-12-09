import pandas
from scanpy.datasets import pbmc68k_reduced
from numpy import max, min

from liana.method import cellphonedb, singlecellsignalr as sca, \
    natmi, connectome, logfc, geometric_mean, cellchat, rank_aggregate

adata = pbmc68k_reduced()
expected_shape = adata.shape


def test_cellchat():
    cellchat(adata, groupby='bulk_labels', use_raw=True, n_perms=2)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'lr_probs' in liana_res.columns
    assert 'cellchat_pvals' in liana_res.columns
    assert max(liana_res[(liana_res.ligand == "TIMP1")].lr_probs) == 0.20561589810421071


def test_cellphonedb():
    cellphonedb(adata, groupby='bulk_labels', use_raw=True, n_perms=2)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'lr_means' in liana_res.columns
    assert 'cellphone_pvals' in liana_res.columns
    assert max(liana_res[(liana_res.ligand == "TIMP1")].lr_means) == 2.134743630886078


def test_geometric_mean():
    geometric_mean(adata, groupby='bulk_labels', use_raw=True, n_perms=2)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'lr_gmeans' in liana_res.columns
    assert 'pvals' in liana_res.columns
    assert max(liana_res[(liana_res.ligand == "TIMP1")].lr_gmeans) == 2.126363309240465


def test_natmi():
    natmi(adata, groupby='bulk_labels', use_raw=True)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'expr_prod' in liana_res.columns
    assert 'spec_weight' in liana_res.columns
    assert max(liana_res[(liana_res.ligand == "TIMP1")].spec_weight) == 0.0604750001773605
    assert max(liana_res[(liana_res.ligand == "TIMP1")].expr_prod) == 4.521420922884062


def test_sca():
    sca(adata, groupby='bulk_labels', use_raw=True)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'lrscore' in liana_res.columns
    assert max(liana_res[(liana_res.ligand == "TIMP1")].lrscore) == 0.781133536891427


def test_logfc():
    logfc(adata, groupby='bulk_labels', use_raw=True)

    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'lr_logfc' in liana_res.columns
    assert max(liana_res[(liana_res.ligand == "TIMP1")].lr_logfc) == 1.4352725744247437


def test_connectome():
    connectome(adata, groupby='bulk_labels', use_raw=True)
    assert adata.shape == expected_shape
    assert 'liana_res' in adata.uns.keys()

    liana_res = adata.uns['liana_res']
    assert isinstance(liana_res, pandas.DataFrame)

    assert 'expr_prod' in liana_res.columns
    assert 'scaled_weight' in liana_res.columns
    assert max(liana_res[(liana_res.ligand == "TIMP1")].scaled_weight) == 0.9669451713562012
    assert max(liana_res[(liana_res.ligand == "TIMP1")].expr_prod) == 4.521420922884062


def test_with_all_lrs():
    natmi(adata, groupby='bulk_labels', use_raw=True, return_all_lrs=True)
    lr_all = adata.uns['liana_res']
    assert lr_all.shape == (4200, 15)
    assert all(lr_all[~lr_all.lrs_to_keep][natmi.magnitude] == min(lr_all[natmi.magnitude])) is True