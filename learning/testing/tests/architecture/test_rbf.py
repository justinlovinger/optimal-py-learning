###############################################################################
# The MIT License (MIT)
#
# Copyright (c) 2017 Justin Lovinger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import random

import pytest
import numpy

from learning import datasets, validation
from learning.architecture import rbf

from learning.testing import helpers


def test_RBF_activate_high_distance_scale_by_similarity():
    """RBF may return nan if sum of similarities == 0 and it scales by similarity."""
    from learning import SOM

    random.seed(0)
    numpy.random.seed(0)

    clustering_model = SOM(1, 2, neighborhood=0)
    clustering_model.logging = False
    model = model = rbf.RBF(
        1,
        2,
        1,
        variance=1.0,
        scale_by_similarity=True,
        clustering_model=clustering_model)
    model._pre_train(numpy.array([[0], [1]]), numpy.array([[0], [1]]))
    assert helpers.approx_equal(model._clustering_model.activate([0]), [0, 1])

    assert not numpy.isnan(model.activate(numpy.array([1000.]))).any()
    assert helpers.approx_equal(model._similarity_tensor, [0.5, 0.5])

    assert not numpy.isnan(model.activate(numpy.array([[0.], [1000.]]))).any()
    assert helpers.approx_equal(model._similarity_tensor,
                                [[0.73105858, 0.26894142], [0.5, 0.5]])


def test_RBF_reset():
    attrs = random.randint(1, 10)
    neurons = random.randint(1, 10)
    outs = random.randint(1, 10)

    model = rbf.RBF(attrs, neurons, outs)
    model_2 = rbf.RBF(attrs, neurons, outs)

    # Resetting different with the same seed should give the same model
    prev_seed = random.randint(0, 2**32-1)

    try:
        random.seed(0)
        numpy.random.seed(0)
        model.reset()

        random.seed(0)
        numpy.random.seed(0)
        model_2.reset()

        assert model.serialize() == model_2.serialize()
    finally:
        random.seed(prev_seed)
        numpy.random.seed(prev_seed)


########################
# Integration tests
########################
def test_rbf():
    # Run for a couple of iterations
    # assert that new error is less than original
    model = rbf.RBF(2, 4, 2, scale_by_similarity=True)
    dataset = datasets.get_xor()

    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=10)
    assert validation.get_error(model, *dataset) < error


def test_rbf_cluster_incrementally():
    # Run for a couple of iterations
    # assert that new error is less than original
    model = rbf.RBF(2, 4, 2, scale_by_similarity=True, cluster_incrementally=True)
    dataset = datasets.get_xor()

    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=10)
    assert validation.get_error(model, *dataset) < error


@pytest.mark.slowtest
def test_rbf_convergence():
    # Run until convergence
    # assert that network can converge
    model = rbf.RBF(2, 4, 2, scale_by_similarity=True)
    dataset = datasets.get_xor()

    model.train(*dataset, retries=5, error_break=0.002)
    assert validation.get_error(model, *dataset) <= 0.02


########################
# Derivative
########################
def test_rbf_obj_and_obj_jac_match():
    """obj and obj_jac functions should return the same obj value."""
    attrs = random.randint(1, 10)
    outs = random.randint(1, 10)
    model = rbf.RBF(attrs, random.randint(1, 10), outs)

    dataset = datasets.get_random_regression(10, attrs, outs)

    # Don't use exactly the same parameters, to ensure obj functions are actually
    # using the given parameters
    parameters = random.uniform(-1.0, 1.0) * rbf._flatten_weights(
        model._weight_matrix, model._bias_vec)
    assert helpers.approx_equal(
        model._get_obj(parameters, dataset[0], dataset[1]),
        model._get_obj_jac(parameters, dataset[0], dataset[1])[0])


def test_rbf_jacobian_scale_by_similarity():
    _check_jacobian(lambda a, n, o: rbf.RBF(a, n, o, scale_by_similarity=True))


def test_rbf_jacobian():
    _check_jacobian(
        lambda a, n, o: rbf.RBF(a, n, o, scale_by_similarity=False))

def _check_jacobian(make_model_func):
    attrs = random.randint(1, 10)
    outs = random.randint(1, 10)

    model = make_model_func(attrs, random.randint(1, 10), outs)
    inp_matrix, tar_matrix = datasets.get_random_regression(10, attrs, outs)

    # Test jacobian of error function
    f = lambda xk: model._get_obj(xk, inp_matrix, tar_matrix)
    df = lambda xk: model._get_obj_jac(xk, inp_matrix, tar_matrix)[1]

    helpers.check_gradient(
        f,
        df,
        f_arg_tensor=rbf._flatten_weights(model._weight_matrix,
                                          model._bias_vec),
        f_shape='scalar')
