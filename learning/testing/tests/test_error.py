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


def test_cross_entropy_zero_in_vec_a():
    """Should not raise error when zeros are in vec_a.

    Because CE takes log of first vector, it can have issues with vectors containing 0s.
    """
    error_func = error.CrossEntropyError()
    assert error_func(numpy.array([0., 0., 1.]), numpy.array([0., 0., 1.])) == 0
    assert error_func(numpy.array([1., 0., 1.]), numpy.array([0., 0., 1.])) == 0


def test_cross_entropy_error_on_negative():
    """CrossEntropy does not take negative values in vec_a."""
    error_func = error.CrossEntropyError()

    with pytest.raises(FloatingPointError):
        assert error_func(numpy.array([-1., 1.]), numpy.array([0., 1.]))


def test_mse_derivative():
    check_error_gradient(error.MeanSquaredError())


def test_cross_entropy_derivative():
    check_error_gradient(error.CrossEntropyError())


def test_cross_entropy_derivative_equals():
    """Should not raise error or return nan, when both inputs match.

    Because the derivative includes a division, this could occur.
    """
    assert (list(error.CrossEntropyError().derivative(
        numpy.array([0., 1.]), numpy.array([0., 1.]))[1]) == [0., -0.5])


def check_error_gradient(error_func):
    vec_length = random.randint(1, 10)

    vec_b = numpy.random.random(vec_length)
    helpers.check_gradient(
        lambda X: error_func(X, vec_b),
        lambda X: error_func.derivative(X, vec_b)[1],
        f_arg_tensor=numpy.random.random(vec_length),
        f_shape='scalar')


#############################
# Penalty Functions
#############################
def test_L1Penalty_jacobian():
    penalty_func = error.L1Penalty(penalty_weight=random.uniform(0.0, 2.0))
    helpers.check_gradient(penalty_func, penalty_func.derivative)


def test_L2Penalty_jacobian():
    penalty_func = error.L2Penalty(penalty_weight=random.uniform(0.0, 2.0))
    helpers.check_gradient(penalty_func, penalty_func.derivative)
