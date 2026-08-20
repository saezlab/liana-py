from liana.multi import to_tensor_c2c


def test_to_tensor_c2c(liana_res_by_sample):
    """Test to_tensor_c2c."""
    import cell2cell as c2c

    liana_dict = to_tensor_c2c(liana_res=liana_res_by_sample,
                               sample_key='sample',
                               score_key='specificity_rank',
                               return_dict=True
                               )
    assert isinstance(liana_dict, dict)

    tensor = to_tensor_c2c(liana_res=liana_res_by_sample,
                           sample_key='sample',
                           score_key='specificity_rank')
    assert isinstance(tensor, c2c.tensor.tensor.PreBuiltTensor)
    assert tensor.sparsity_fraction()==0.0
