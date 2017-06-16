"""Test cases for learning/error.py."""
import random

import numpy

from learning import error

from learning.testing import helpers

def test_mse_derivative():
    check_error_gradient(error.MSE())

def test_cross_entropy_derivative():
    check_error_gradient(error.CrossEntropy())

def check_error_gradient(error_func):
    vec_length = random.randint(1, 10)

    vec_b = numpy.random.random(vec_length)
    helpers.check_gradient(
        lambda X: error_func(X, vec_b),
        lambda X: error_func.derivative(X, vec_b)[1],
        inputs=numpy.random.random(vec_length),
        f_shape='scalar'
    )
