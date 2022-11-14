import pathlib

from unittest import TestCase
from liana.method import rank_aggregate
from liana.method.sc._rank_aggregate import AggregateClass
from scanpy.datasets import pbmc68k_reduced
from pandas import read_csv
from pandas.testing import assert_frame_equal

test_path = pathlib.Path(__file__).parent


def test_consensus():
    assert isinstance(rank_aggregate, AggregateClass)
    assert rank_aggregate.magnitude == 'magnitude_rank'
    assert rank_aggregate.specificity == 'specificity_rank'
    assert rank_aggregate.steady == 'steady_rank'
    assert rank_aggregate.method_name == 'Rank_Aggregate'


def test_consensus_specs():
    steady_specs = {'CellPhoneDB': ('pvals', True),
                    'Connectome': ('scaled_weight', False),
                    'log2FC': ('lr_logfc', False),
                    'NATMI': ('spec_weight', False),
                    'SingleCellSignalR': ('lrscore', False)
                    }
    TestCase().assertDictEqual(rank_aggregate.steady_specs, steady_specs)

    magnitude_specs = {'CellPhoneDB': ('lr_means', False),
                       'Connectome': ('expr_prod', False),
                       'NATMI': ('expr_prod', False),
                       'SingleCellSignalR': ('lrscore', False)
                       }

    TestCase().assertDictEqual(rank_aggregate.magnitude_specs, magnitude_specs)


def test_consensus_res():
    adata = pbmc68k_reduced()
    lr_res = rank_aggregate(adata, groupby='bulk_labels', use_raw=True, n_perms=2, copy=True)
    lr_exp = read_csv(test_path.joinpath("data/aggregate_rank_rest.csv"), index_col=0)

    assert_frame_equal(lr_res, lr_exp, check_dtype=False,
                       check_exact=False, check_less_precise=True)
