import anndata

from liana.method import cellphonedb, singlecellsignalr as sca, natmi, connectome, logfc
from scanpy.datasets import pbmc68k_reduced

adata = pbmc68k_reduced()


def test_cellphonedb():
    test_cellphonedb = cellphonedb(adata, groupby='bulk_labels', use_raw=True, n_perms=2)
    assert isinstance(test_cellphonedb, anndata.AnnData)
    assert 'liana_res' in test_cellphonedb.uns.keys()


def test_natmi():
    test_natmi = natmi(adata, groupby='bulk_labels', use_raw=True)
    assert isinstance(test_natmi, anndata.AnnData)
    assert 'liana_res' in test_natmi.uns.keys()


def test_sca():
    test_sca = sca(adata, groupby='bulk_labels', use_raw=True)
    assert isinstance(test_sca, anndata.AnnData)
    assert 'liana_res' in test_sca.uns.keys()


def test_logfc():
    test_logfc = logfc(adata, groupby='bulk_labels', use_raw=True)
    assert isinstance(test_logfc, anndata.AnnData)
    assert 'liana_res' in test_logfc.uns.keys()


def test_connectome():
    test_connectome = connectome(adata, groupby='bulk_labels', use_raw=True)
    assert isinstance(test_connectome, anndata.AnnData)
    assert 'liana_res' in test_connectome.uns.keys()
