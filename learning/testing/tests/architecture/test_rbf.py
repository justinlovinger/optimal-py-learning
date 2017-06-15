import random

import pytest

from learning.architecture import rbf
from learning.data import datasets

from learning.testing import helpers

########################
# Integration tests
########################
def test_rbf():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = rbf.RBF(2, 4, 2, scale_by_similarity=True)
    pat = datasets.get_xor()

    error = nn.avg_mse(*pat)
    nn.train(*pat, iterations=10)
    assert nn.avg_mse(*pat) < error


pytest.mark.slowtest()
def test_rbf_convergence():
    # Run until convergence
    # assert that network can converge
    nn = rbf.RBF(2, 4, 2, scale_by_similarity=True)
    pat = datasets.get_xor()

    nn.train(*pat, retries=5, error_break=0.002)
    assert nn.avg_mse(*pat) <= 0.02

def test_rbf_obj_and_obj_jac_match():
    """obj and obj_jac functions should return the same obj value."""
    attrs = random.randint(1, 10)
    outs = random.randint(1, 10)
    model = rbf.RBF(attrs, random.randint(1, 10), outs)

    dataset = datasets.get_random_regression(10, attrs, outs)

    parameters = model._weight_matrix.ravel()
    assert helpers.approx_equal(model._get_obj(parameters, dataset[0], dataset[1]),
                                model._get_obj_jac(parameters, dataset[0], dataset[1])[0])

def test_rbf_jacobian_scale_by_similarity():
    _check_jacobian(lambda a, n, o: rbf.RBF(a, n, o, scale_by_similarity=True))

def test_rbf_jacobian():
    _check_jacobian(lambda a, n, o: rbf.RBF(a, n, o, scale_by_similarity=False))

def _check_jacobian(make_model_func):
    attrs = random.randint(1, 10)
    outs = random.randint(1, 10)

    model = make_model_func(attrs, random.randint(1, 10), outs)
    inp_matrix, tar_matrix = datasets.get_random_regression(10, attrs, outs)

    # Test jacobian of error function
    f = lambda xk: model._get_obj(xk, inp_matrix, tar_matrix)
    df = lambda xk: model._get_obj_jac(xk, inp_matrix, tar_matrix)[1]

    helpers.check_gradient(f, df, inputs=model._weight_matrix.ravel(), f_shape='scalar')
