import unittest

import anndata

from liana import cellphonedb, singlecellsignalr as sca, natmi, connectome, logfc
from scanpy.datasets import pbmc68k_reduced

adata = pbmc68k_reduced()


class TestMethods(unittest.TestCase):
    def test_cellphonedb(self):
        test_cellphonedb = cellphonedb(adata, groupby='bulk_labels', use_raw=True, n_perms=2)
        self.assertIsInstance(test_cellphonedb, anndata.AnnData)
        self.assertIn('liana_res', adata.uns)

    def test_natmi(self):
        test_natmi = natmi(adata, groupby='bulk_labels', use_raw=True)
        self.assertIsInstance(test_natmi, anndata.AnnData)
        self.assertIn('liana_res', adata.uns)

    def test_sca(self):
        test_sca = sca(adata, groupby='bulk_labels', use_raw=True)
        self.assertIsInstance(test_sca, anndata.AnnData)
        self.assertIn('liana_res', adata.uns)

    def test_logfc(self):
        test_logfc = logfc(adata, groupby='bulk_labels', use_raw=True)
        self.assertIsInstance(test_logfc, anndata.AnnData)
        self.assertIn('liana_res', adata.uns)

    def test_connectome(self):
        test_connectome = connectome(adata, groupby='bulk_labels', use_raw=True)
        self.assertIsInstance(test_connectome, anndata.AnnData)
        self.assertIn('liana_res', adata.uns)


if __name__ == '__main__':
    unittest.main()
