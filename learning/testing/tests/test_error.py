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
"""Test cases for learning/error.py."""
import random

import numpy
import pytest

from learning import error

from learning.testing import helpers


#########################
# MSE
#########################
def test_mse_vector():
    error.MeanSquaredError()(numpy.array([1, 1]), numpy.array([1, 1])) == 0
    error.MeanSquaredError()(numpy.array([1, 0]), numpy.array([1, 1])) == 0.5
    error.MeanSquaredError()(numpy.array([0, 0]), numpy.array([1, 1])) == 1
    error.MeanSquaredError()(numpy.array([0.5, 0.5]), numpy.array([1, 1])) == 0.25
    error.MeanSquaredError()(numpy.array([0, 0.5]), numpy.array([0, 1])) == 0.125

    error.MeanSquaredError()(numpy.array([1, 1, 1]), numpy.array([0, 0, 0])) == 1

    error.MeanSquaredError()(numpy.array([1, 1, 1, 0]), numpy.array([0, 0, 1, 1])) == 0.75


def test_mse_matrix():
    error.MeanSquaredError()(numpy.array([[1, 1], [1, 1]]), numpy.array([[0, 0], [0, 0]])) == 1
    error.MeanSquaredError()(numpy.array([[1, 1], [0, 0]]), numpy.array([[1, 0], [1, 0]])) == 0.5
    error.MeanSquaredError()(numpy.array([[1, 1], [1, 0]]), numpy.array([[0, 0], [1, 1]])) == 0.75
    error.MeanSquaredError()(numpy.array([[0, 1], [0, 0]]), numpy.array([[0, 0.5], [0.5, 0]])) == 0.25
    error.MeanSquaredError()(numpy.array([[0, 0], [0, 0]]), numpy.array([[0, 0], [0, 0]])) == 0


def test_mse_derivative_vector():
    check_error_gradient(error.MeanSquaredError(), tensor_d=1)


def test_mse_derivative_matrix():
    check_error_gradient(error.MeanSquaredError(), tensor_d=2)


def test_mse_derivative_error_equals_call_error_vec():
    check_derivative_error_equals_call_error(
        error.MeanSquaredError(), tensor_d=1)


def test_mse_derivative_error_equals_call_error_matrix():
    check_derivative_error_equals_call_error(
        error.MeanSquaredError(), tensor_d=2)


#########################
# Cross Entropy
#########################
def test_cross_entropy_vector():
    assert error.CrossEntropyError()(numpy.array([0.1, 1.]),
                                     numpy.array([0., 1.])) == 0
    assert error.CrossEntropyError()(numpy.array([0.1, 1. / numpy.e]),
                                     numpy.array([0., 1.])) == 1


def test_cross_entropy_matrix():
    assert error.CrossEntropyError()(numpy.array([[0.1, 1.], [1., 0.1]]),
                                     numpy.array([[0., 1.], [1., 0.]])) == 0
    assert error.CrossEntropyError()(numpy.array([[0.1, 1. / numpy.e], [1. / numpy.e, 0.1]]),
                                     numpy.array([[0., 1.], [1., 0.]])) == 1
    assert error.CrossEntropyError()(numpy.array([[0.1, 1.], [1. / numpy.e, 0.1]]),
                                     numpy.array([[0., 1.], [1., 0.]])) == 0.5


def test_cross_entropy_zero_in_tensor_a():
    """Should not raise error when zeros are in tensor_a.

    Because CE takes log of first vector, it can have issues with vectors containing 0s.
    """
    error_func = error.CrossEntropyError()
    assert error_func(numpy.array([0., 0., 1.]), numpy.array([0., 0., 1.])) == 0
    assert error_func(numpy.array([1., 0., 1.]), numpy.array([0., 0., 1.])) == 0
    assert error_func(numpy.array([0., 1. / numpy.e]), numpy.array([0., 1.])) == 1


def test_cross_entropy_error_on_negative():
    """CrossEntropy does not take negative values in tensor_a."""
    error_func = error.CrossEntropyError()

    with pytest.raises(FloatingPointError):
        assert error_func(numpy.array([-1., 1.]), numpy.array([0., 1.]))


def test_cross_entropy_derivative_vector():
    check_error_gradient(error.CrossEntropyError(), tensor_d=1)


def test_cross_entropy_derivative_matrix():
    check_error_gradient(error.CrossEntropyError(), tensor_d=2)


def test_cross_entropy_derivative_equal_tensors():
    """Should not raise error or return nan, when both inputs match.

    Because the derivative includes a division, this could occur.
    """
    assert (list(error.CrossEntropyError().derivative(
        numpy.array([0., 1.]), numpy.array([0., 1.]))[1]) == [0., -1])


def test_cross_entropy_derivative_error_equals_call_error_vec():
    check_derivative_error_equals_call_error(
        error.CrossEntropyError(), tensor_d=1)


def test_cross_entropy_derivative_error_equals_call_error_matrix():
    check_derivative_error_equals_call_error(
        error.CrossEntropyError(), tensor_d=2)


#############################
# Penalty Functions
#############################
def test_L1Penalty_jacobian():
    penalty_func = error.L1Penalty(penalty_weight=random.uniform(0.0, 2.0))
    helpers.check_gradient(penalty_func, penalty_func.derivative)


def test_L2Penalty_jacobian():
    penalty_func = error.L2Penalty(penalty_weight=random.uniform(0.0, 2.0))
    helpers.check_gradient(penalty_func, penalty_func.derivative)


#############################
# Helpers
#############################
def check_error_gradient(error_func, tensor_d=1):
    tensor_shape = [random.randint(1, 10) for _ in range(tensor_d)]

    tensor_b = numpy.random.random(tensor_shape)
    helpers.check_gradient(
        lambda X: error_func(X, tensor_b),
        lambda X: error_func.derivative(X, tensor_b)[1],
        f_arg_tensor=numpy.random.random(tensor_shape),
        f_shape='scalar')


def check_derivative_error_equals_call_error(error_func, tensor_d=1):
    tensor_shape = [random.randint(1, 10) for _ in range(tensor_d)]
    tensor_a = numpy.random.random(tensor_shape)
    tensor_b = numpy.random.random(tensor_shape)

    assert error_func(tensor_a, tensor_b) == error_func.derivative(tensor_a, tensor_b)[0]
