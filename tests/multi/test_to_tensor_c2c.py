from pandas import DataFrame

from liana.multisample import to_tensor_c2c


def test_to_tensor_c2c(liana_res_by_sample: DataFrame) -> None:
    """Test to_tensor_c2c."""
    from cell2cell.tensor.tensor import PreBuiltTensor

    liana_dict = to_tensor_c2c(
        liana_res=liana_res_by_sample, sample_key="sample", score_key="specificity_rank", return_dict=True
    )
    assert isinstance(liana_dict, dict)

    tensor = to_tensor_c2c(liana_res=liana_res_by_sample, sample_key="sample", score_key="specificity_rank")
    assert isinstance(tensor, PreBuiltTensor)
    # bound back to the class itself: the intersection type `isinstance` narrows to
    # is not what `untyped_calls_exclude` matches against
    prebuilt: PreBuiltTensor = tensor
    assert prebuilt.sparsity_fraction() == 0.0
