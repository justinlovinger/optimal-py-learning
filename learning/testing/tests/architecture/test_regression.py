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

from learning import LinearRegressionModel
from learning.data import datasets
from learning.testing import helpers


######################################
# LinearRegressionModel
######################################
def test_LinearRegressionModel_jacobian():
    _check_jacobian(lambda a, o: LinearRegressionModel(a, o,))


def test_LinearRegressionModel_equation_derivative():
    _check_equation_derivative(lambda a, o: LinearRegressionModel(a, o,))


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


def _check_equation_derivative(make_model_func):
    attrs = random.randint(1, 10)
    outs = random.randint(1, 10)

    # Test jacobian of equation output, with regard to weight matrix
    model = make_model_func(attrs, outs)
    input_vec = numpy.random.random(attrs)

    def f(flat_weights):
        model._weight_matrix = flat_weights.reshape(model._weight_matrix.shape)
        return model._equation_output(input_vec)

    def df(flat_weights):
        model._weight_matrix = flat_weights.reshape(model._weight_matrix.shape)
        return model._equation_derivative(input_vec)

    helpers.check_gradient(
        f,
        df,
        inputs=(
            numpy.random.random(model._weight_matrix.shape) * 2 - 1).ravel(),
        f_shape='jac')
