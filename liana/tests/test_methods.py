import pandas

from liana.method import cellphonedb, singlecellsignalr as sca, \
    natmi, connectome, logfc, geometric_mean, cellchat
from scanpy.datasets import pbmc68k_reduced

adata = pbmc68k_reduced()
expected_shape = adata.shape


def test_cellchat():
    cellchat(adata, groupby='bulk_labels', use_raw=True, n_perms=2)
    assert 'liana_res' in adata.uns.keys()
    assert isinstance(adata.uns['liana_res'], pandas.DataFrame)
    assert 'lr_probs' in adata.uns['liana_res'].columns
    assert 'pvals' in adata.uns['liana_res'].columns
    assert adata.shape == expected_shape
    # check values for a specific row


def test_cellphonedb():
    cellphonedb(adata, groupby='bulk_labels', use_raw=True, n_perms=2)
    assert 'liana_res' in adata.uns.keys()
    assert isinstance(adata.uns['liana_res'], pandas.DataFrame)
    assert 'lr_means' in adata.uns['liana_res'].columns
    assert 'pvals' in adata.uns['liana_res'].columns
    assert adata.shape == expected_shape


def test_geometric_mean():
    geometric_mean(adata, groupby='bulk_labels', use_raw=True, n_perms=2)
    assert 'liana_res' in adata.uns.keys()
    assert isinstance(adata.uns['liana_res'], pandas.DataFrame)
    assert 'lr_gmeans' in adata.uns['liana_res'].columns
    assert 'pvals' in adata.uns['liana_res'].columns
    assert adata.shape == expected_shape


def test_natmi():
    natmi(adata, groupby='bulk_labels', use_raw=True)
    assert 'liana_res' in adata.uns.keys()
    assert isinstance(adata.uns['liana_res'], pandas.DataFrame)
    assert 'expr_prod' in adata.uns['liana_res'].columns
    assert 'spec_weight' in adata.uns['liana_res'].columns
    assert adata.shape == expected_shape


def test_sca():
    sca(adata, groupby='bulk_labels', use_raw=True)
    assert 'liana_res' in adata.uns.keys()
    assert isinstance(adata.uns['liana_res'], pandas.DataFrame)
    assert 'lrscore' in adata.uns['liana_res'].columns
    assert adata.shape == expected_shape


def test_logfc():
    logfc(adata, groupby='bulk_labels', use_raw=True)
    assert 'liana_res' in adata.uns.keys()
    assert isinstance(adata.uns['liana_res'], pandas.DataFrame)
    assert 'lr_logfc' in adata.uns['liana_res'].columns
    assert adata.shape == expected_shape


def test_connectome():
    connectome(adata, groupby='bulk_labels', use_raw=True)
    assert 'liana_res' in adata.uns.keys()
    assert isinstance(adata.uns['liana_res'], pandas.DataFrame)
    assert 'expr_prod' in adata.uns['liana_res'].columns
    assert 'scaled_weight' in adata.uns['liana_res'].columns
    assert adata.shape == expected_shape
