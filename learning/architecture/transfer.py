"""Transfer function layers."""

import numpy
import math


def tanh(x):
    """Sigmoid like function using tanh"""
    return numpy.tanh(x)

def dtanh(y):
    """Derivative of sigmoid above"""
    return 1.0 - y**2

def gaussian(x, variance=1.0):
    return numpy.exp(-(x**2/variance))

def dgaussian(x, y, variance=1.0):
    return -2.0*x*y / variance

def relu(x):
    """Return ln(1 + e^x) for each input value."""
    return numpy.log(1 + numpy.e**x)

def drelu(x):
    """Return the derivative of the softplus relu function for y."""
    # NOTE: Can be optimized by caching numpy.e**(x) and returning e^x / (e^x + 1)
    return 1.0 / (1.0 + numpy.e**(-x))

def softmax(x):
    """Return the softmax of vector x."""
    exp_ = numpy.exp(x)
    return exp_ / numpy.sum(exp_)

def dsoftmax(y):
    """Return the derivative of the softmax function for y."""
    # see http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
    # Compute matrix J, n x n, with y_i(1 - y_j) on the diagonals
    # and - y_i y_j on the non-diagonals
    # When getting erros multiply by error vector (J \vec{e})

    # Start with - y_i y_j matrix, then replace diagonal with y_i(1 - y_j)
    jacobian = -y[:, None] * y
    jacobian[numpy.diag_indices(y.shape[0])] = y*(1 - y)
    return jacobian
