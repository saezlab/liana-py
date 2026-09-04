from typing import TYPE_CHECKING

import pytest
from pandas import DataFrame

from liana.multisample import to_tensor_c2c

if TYPE_CHECKING:
    from cell2cell.tensor.tensor import PreBuiltTensor


def test_to_tensor_c2c(liana_res_by_sample: DataFrame) -> None:
    """Test to_tensor_c2c."""
    # `cell2cell` is an optional extra, and it imports `scanpy.readwrite`, which scanpy has since moved to `scanpy.io`.
    tensor_cls = pytest.importorskip(
        "cell2cell.tensor.tensor", reason="cell2cell is not importable here"
    ).PreBuiltTensor

    liana_dict = to_tensor_c2c(
        liana_res=liana_res_by_sample, sample_key="sample", score_key="specificity_rank", return_dict=True
    )
    assert isinstance(liana_dict, dict)

    tensor = to_tensor_c2c(liana_res=liana_res_by_sample, sample_key="sample", score_key="specificity_rank")
    assert isinstance(tensor, tensor_cls)
    # `untyped_calls_exclude` does not match the intersection type `isinstance` narrows to
    prebuilt: PreBuiltTensor = tensor
    assert prebuilt.sparsity_fraction() == 0.0
