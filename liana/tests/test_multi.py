from ..multi import to_tensor_c2c
from ..testing import sample_lrs
import cell2cell as c2c


def test_to_tensor_c2c():
    """Test to_tensor_c2c."""
    liana_res = sample_lrs(by_sample=True)
    liana_res = liana_res.drop_duplicates(['source', 'target', 'ligand_complex', 'receptor_complex'])
    
    liana_dict = to_tensor_c2c(liana_res=liana_res,
                               sample_key='sample',
                               score_key='specificity_rank',
                               return_dict=True
                               )
    assert isinstance(liana_dict, dict)
    
    tensor = to_tensor_c2c(liana_res=liana_res,
                           sample_key='sample',
                           score_key='specificity_rank')
    assert isinstance(tensor, c2c.tensor.tensor.PreBuiltTensor)
    assert tensor.sparsity_fraction()==0.0
