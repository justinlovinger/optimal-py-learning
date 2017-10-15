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

import numpy
import pytest

from learning import validation
from learning import LinearRegressionModel
from learning.data import datasets
from learning.testing import helpers


######################################
# LinearRegressionModel
######################################
def test_LinearRegressionModel():
    # Run for a couple of iterations
    # assert that new error is less than original
    model = LinearRegressionModel(2, 2)
    # NOTE: We use and instead of xor, because xor is non-linear
    dataset = datasets.get_and()

    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=10)
    assert validation.get_error(model, *dataset) < error


@pytest.mark.slowtest
def test_LinearRegressionModel_convergence():
    # Run until convergence
    # assert that model can converge
    model = LinearRegressionModel(2, 1)
    # NOTE: We use make plus dataset, so linear model can fully converge
    input_matrix = numpy.random.random((10, 2))
    target_matrix = numpy.sum(input_matrix, axis=1)[:, None]

    error = validation.get_error(model, input_matrix, target_matrix)
    model.train(input_matrix, target_matrix, iterations=10)
    assert validation.get_error(model, input_matrix, target_matrix) <= 0.02


def test_LinearRegressionModel_jacobian():
    _check_jacobian(lambda a, o: LinearRegressionModel(a, o,))


######################################
# Helpers
######################################
def _check_jacobian(make_model_func):
    attrs = random.randint(1, 10)
    outs = random.randint(1, 10)

    model = make_model_func(attrs, outs)
    inp_matrix, tar_matrix = datasets.get_random_regression(10, attrs, outs)

    # Test jacobian of error function
    f = lambda xk: model._get_obj(xk, inp_matrix, tar_matrix)
    df = lambda xk: model._get_obj_jac(xk, inp_matrix, tar_matrix)[1]

    helpers.check_gradient(
        f, df, inputs=model._weight_matrix.ravel(), f_shape='scalar')
