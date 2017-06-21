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
    error_func = error.CrossEntropy()
    assert error_func(numpy.array([0., 0., 1.]), numpy.array([0., 0., 1.])) == 0
    assert error_func(numpy.array([1., 0., 1.]), numpy.array([0., 0., 1.])) == 0

def test_cross_entropy_error_on_negative():
    """CrossEntropy does not take negative values in vec_a."""
    error_func = error.CrossEntropy()

    with pytest.raises(FloatingPointError):
        assert error_func(numpy.array([-1., 1.]), numpy.array([0., 1.]))

def test_mse_derivative():
    check_error_gradient(error.MSE())

def test_cross_entropy_derivative():
    check_error_gradient(error.CrossEntropy())

def test_cross_entropy_derivative_equals():
    """Should not raise error or return nan, when both inputs match.

    Because the derivative includes a division, this could occur.
    """
    assert (
        list(error.CrossEntropy().derivative(
            numpy.array([0., 1.]), numpy.array([0., 1.])
        )[1])
        == [0., -0.5]
    )

def check_error_gradient(error_func):
    vec_length = random.randint(1, 10)

    vec_b = numpy.random.random(vec_length)
    helpers.check_gradient(
        lambda X: error_func(X, vec_b),
        lambda X: error_func.derivative(X, vec_b)[1],
        inputs=numpy.random.random(vec_length),
        f_shape='scalar'
    )
