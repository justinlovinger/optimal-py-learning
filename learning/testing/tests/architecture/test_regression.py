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

from learning import datasets, validation, error, LinearRegressionModel

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
    model = LinearRegressionModel(2, 2)
    # NOTE: We use and instead of xor, because xor is non-linear
    dataset = datasets.get_and()

    model.train(*dataset)
    # NOTE: This linear model cannot achieve 0 MSE
    assert validation.get_error(model, *dataset) <= 0.1


def test_LinearRegressionModel_jacobian():
    _check_jacobian(lambda a, o: LinearRegressionModel(a, o))


def test_LinearRegressionModel_jacobian_l1_penalty():
    _check_jacobian(lambda a, o: LinearRegressionModel(
        a, o, penalty_func=error.L1Penalty(
            penalty_weight=random.uniform(0.0, 2.0))))


def test_LinearRegressionModel_jacobian_l2_penalty():
    _check_jacobian(lambda a, o: LinearRegressionModel(
        a, o, penalty_func=error.L2Penalty(
            penalty_weight=random.uniform(0.0, 2.0))))


def test_LinearRegressionModel_get_obj_equals_get_obj_jac():
    _check_get_obj_equals_get_obj_jac(lambda a, o: LinearRegressionModel(a, o))


def test_LinearRegressionModel_get_obj_equals_get_obj_jac_l1_penalty():
    _check_get_obj_equals_get_obj_jac(lambda a, o: LinearRegressionModel(
        a, o, penalty_func=error.L1Penalty(
            penalty_weight=random.uniform(0.0, 2.0))))


def test_LinearRegressionModel_get_obj_equals_get_obj_jac_l2_penalty():
    _check_get_obj_equals_get_obj_jac(lambda a, o: LinearRegressionModel(
        a, o, penalty_func=error.L2Penalty(
            penalty_weight=random.uniform(0.0, 2.0))))


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


def _check_get_obj_equals_get_obj_jac(make_model_func):
    attrs = random.randint(1, 10)
    outs = random.randint(1, 10)

    inp_matrix, tar_matrix = datasets.get_random_regression(10, attrs, outs)
    model = make_model_func(attrs, outs)
    flat_weights = model._weight_matrix.ravel()

    assert model._get_obj(flat_weights, inp_matrix,
                          tar_matrix) == model._get_obj_jac(
                              flat_weights, inp_matrix, tar_matrix)[0]
