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

from learning import datasets, validation
from learning.architecture import rbf

from learning.testing import helpers


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


@pytest.mark.slowtest
def test_rbf_convergence():
    # Run until convergence
    # assert that network can converge
    model = rbf.RBF(2, 4, 2, scale_by_similarity=True)
    dataset = datasets.get_xor()

    model.train(*dataset, retries=5, error_break=0.002)
    assert validation.get_error(model, *dataset) <= 0.02


def test_rbf_obj_and_obj_jac_match():
    """obj and obj_jac functions should return the same obj value."""
    attrs = random.randint(1, 10)
    outs = random.randint(1, 10)
    model = rbf.RBF(attrs, random.randint(1, 10), outs)

    dataset = datasets.get_random_regression(10, attrs, outs)

    # Don't use exactly the same parameters, to ensure obj functions are actually
    # using the given parameters
    parameters = random.uniform(-1.0, 1.0) * model._flatten_weights(
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
        f_arg_tensor=model._flatten_weights(model._weight_matrix,
                                            model._bias_vec),
        f_shape='scalar')
