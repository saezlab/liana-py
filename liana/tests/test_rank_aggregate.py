import pathlib

from unittest import TestCase
from liana import rank_aggregate
from liana.steady.scores.rank_aggregate import ConsensusClass
from scanpy.datasets import pbmc68k_reduced
from pandas import read_csv
from pandas.testing import assert_frame_equal

test_path = pathlib.Path(__file__).parent


class TestConsensus:
    def test_consensus(self):
        assert isinstance(rank_aggregate, ConsensusClass)
        assert rank_aggregate.magnitude == 'magnitude_rank'
        assert rank_aggregate.specificity == 'specificity_rank'
        assert rank_aggregate.steady == 'steady_rank'
        assert rank_aggregate.method_name == 'Rank_Aggregate'

    def test_consensus_specs(self):
        steady_specs = {'CellPhoneDB': ('pvals', False),
                        'Connectome': ('scaled_weight', True),
                        'log2FC': ('lr_logfc', True),
                        'NATMI': ('spec_weight', True),
                        'SingleCellSignalR': ('lrscore', True)
                        }
        TestCase().assertDictEqual(rank_aggregate.steady_specs, steady_specs)

        magnitude_specs = {'CellPhoneDB': ('lr_means', True),
                           'Connectome': ('expr_prod', True),
                           'NATMI': ('expr_prod', True),
                           'SingleCellSignalR': ('lrscore', True)
                           }

        TestCase().assertDictEqual(rank_aggregate.magnitude_specs, magnitude_specs)

    def test_consensus_res(self):
        adata = pbmc68k_reduced()
        adata = rank_aggregate(adata, groupby='bulk_labels', use_raw=True, n_perms=2)
        lr_res = adata.uns['liana_res']
        lr_exp = read_csv(test_path.joinpath("data/aggregate_rank_rest.csv"), index_col=0)

        assert_frame_equal(lr_res, lr_exp, check_dtype=False,
                           check_exact=False, check_less_precise=True)
