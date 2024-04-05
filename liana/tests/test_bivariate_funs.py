import numpy as np

from liana.method.sp._bivariate._local_functions import _vectorized_pearson, _vectorized_spearman, \
    _vectorized_cosine, _vectorized_jaccard, _masked_spearman, _local_morans, _product, \
    _norm_product

from scipy.sparse import csr_matrix


rng = np.random.default_rng(seed=0)

x_mat = rng.normal(size=(20, 5)).astype(np.float32)
y_mat = rng.normal(size=(20, 5)).astype(np.float32)
weight = csr_matrix(rng.uniform(size=(20, 20)).astype(np.float32))

def _assert_bivariate(function, desired, x_mat, y_mat, weight):
    actual = function(x_mat, y_mat, weight)
    assert actual.shape == (20, 5)
    np.testing.assert_almost_equal(actual[0, :], desired, decimal=5)


def test_pc_vectorized():
    pc_vec_truth = np.array([0.25005114, 0.04262733, -0.00130362, 0.2903336, -0.1236529])
    _assert_bivariate(_vectorized_pearson, pc_vec_truth, x_mat, y_mat, weight)


def test_sp_vectorized():
    sp_vec_truth = np.array([0.23636213, 0.16480759, -0.01487235, 0.22840601, -0.11492937])
    _assert_bivariate(_vectorized_spearman, sp_vec_truth, x_mat, y_mat, weight)


def test_sp_masked():
    sp_masked_truth = np.array([0.23636216, 0.16480756, -0.0148723, 0.22840606, -0.11492944])
    _assert_bivariate(_masked_spearman, sp_masked_truth, x_mat, y_mat, weight.A)  # NOTE the .A is to convert to dense


def test_costine_vectorized():
    cosine_vec_truth = np.array([0.33806977, 0.03215113, 0.0950243, 0.2957758, -0.10259595])
    _assert_bivariate(_vectorized_cosine, cosine_vec_truth, x_mat, y_mat, weight)


def test_vectorized_jaccard():
    jaccard_vec_truth = np.array([0.34295967, 0.35367563, 0.39685577, 0.41780996, 0.30527356])
    _assert_bivariate(_vectorized_jaccard, jaccard_vec_truth, x_mat, y_mat, weight)


# NOTE: spatialdm uses raw counts
def test_morans():
    sp_morans_truth = np.array([-1.54256, 0.64591, 1.30025, 0.55437, -0.77182])
    _assert_bivariate(_local_morans, sp_morans_truth, x_mat, y_mat, weight)


def test_product():
    product_vec_truth = np.array([5.4518123, -0.7268728, 8.350364, 0.53861964, 1.4466602])
    _assert_bivariate(_product, product_vec_truth, x_mat, y_mat, weight.A)


def test_norm_product():
    product_vec_truth = np.array([0.4081537, -0.03988646, 0.42921585, 0.03255661, 0.08895018])
    _assert_bivariate(_norm_product, product_vec_truth, x_mat, y_mat, weight.A)
