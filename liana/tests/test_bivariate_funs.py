import numpy as np

from liana.method.sp._bivariate_funs import _vectorized_pearson, _vectorized_spearman, \
    _vectorized_cosine, _vectorized_jaccard, _masked_pearson, \
         _masked_spearman, _masked_cosine, _masked_jaccard, _local_morans

from scipy.sparse import csr_matrix


rng = np.random.default_rng(seed=0)

xmat = rng.normal(size=(20, 5)).astype(np.float32)
ymat = rng.normal(size=(20, 5)).astype(np.float32)
weight = csr_matrix(rng.uniform(size=(20, 20)).astype(np.float32))


def _assert_bivariate(function, desired, xmat, ymat, weight):
    actual = function(xmat, ymat, weight)
    assert actual.shape == (5, 20)
    np.testing.assert_almost_equal(actual[:,0], desired, decimal=5)
    

def test_pc_vectorized():
    pc_vec_truth = np.array([ 0.45880201,  0.06379483, -0.08202948, -0.07459559, -0.10786078])
    _assert_bivariate(_vectorized_pearson, pc_vec_truth, xmat, ymat, weight)


def test_pc_masked():
    pc_masked_truth = np.array([ 0.25005117,  0.04262732, -0.00130363,  0.2903336 , -0.12365292])
    _assert_bivariate(_masked_pearson, pc_masked_truth, xmat, ymat, weight.A)  # NOTE the .A is to convert to dense

def test_sp_vectorized():
    sp_vec_truth = np.array([ 0.38851726,  0.15194362, -0.02620391, -0.11188834, -0.09263334])
    _assert_bivariate(_vectorized_spearman, sp_vec_truth, xmat, ymat, weight)



def test_sp_masked():
    sp_masked_truth = np.array([0.23636216, 0.16480756, -0.0148723, 0.22840606, -0.11492944])
    _assert_bivariate(_masked_spearman, sp_masked_truth, xmat, ymat, weight.A)  # NOTE the .A is to convert to dense


def test_costine_vectorized():
    cosine_vec_truth = np.array([ 0.31625268,  0.02285767, -0.02824857, -0.01511965, -0.0337257 ])
    _assert_bivariate(_vectorized_cosine, cosine_vec_truth, xmat, ymat, weight)


def test_cosine_masked():
    cosine_masked_truth = np.array([ 0.3380698 ,  0.03215112,  0.09502427,  0.29577583, -0.10259596])
    _assert_bivariate(_masked_cosine, cosine_masked_truth, xmat, ymat, weight.A) # NOTE the .A is to convert to dense


def test_vectorized_jaccard():
    jaccard_vec_truth = np.array([0.4998738 , 0.4665028 , 0.27069882, 0.27474707, 0.35307598])
    _assert_bivariate(_vectorized_jaccard, jaccard_vec_truth, xmat, ymat, weight)


def test_masked_jaccard():
    jac_masked_truth = np.array([0.34295967, 0.35367563, 0.39685577, 0.41780996, 0.30527356])
    _assert_bivariate(_masked_jaccard, jac_masked_truth, xmat, ymat, weight.A) # NOTE the .A is to convert to dense


# TODO double check this with SpatialDM.... (I made a mistake in local only?)
def test_morans():
    sp_morans_truth = np.array([-0.5419496 ,  0.45341554,  3.5817103 , -0.18734339, -2.5889277 ])
    _assert_bivariate(_local_morans, sp_morans_truth, xmat, ymat, weight)
