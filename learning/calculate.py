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
"""Commonly used mathematical functions."""

import numpy


INFINITY = float('inf')


def distance(vec_a, vec_b):
    # TODO: fix so it works with matrix inputs
    diff = numpy.subtract(vec_a, vec_b)
    return numpy.sqrt(diff.dot(diff))


def protvecdiv(vec_a, vec_b):
    """Divide vec_a by vec_b.

    When vec_b_i == 0, return 0 for component i.
    """
    with numpy.errstate(divide='raise', invalid='raise'):
        try:
            # Try to quickly divide vectors
            return vec_a / vec_b
        except FloatingPointError:
            # Fallback to dividing component at a time
            # Slower, but lets us handle divide by 0
            # TODO: Use procedure in preprocess.normalzie instead
            #   Divide, then np_matrix[~ numpy.isfinite(np_matrix)] = 0.0
            result_vec = numpy.zeros(vec_a.shape)
            for i in range(vec_a.shape[0]):
                try:
                    result_vec[i] = vec_a[i] / vec_b[i]
                except FloatingPointError:
                    pass  # Already 0 from numpy.zeros
            return result_vec


#####################################
# Common math and transfer functions
#####################################
def logit(x):
    """Return logistic function, f(x) = 1 / (1 + e^{-x})."""
    return 1.0 / (1.0 + numpy.exp(-x))


def dlogit(x):
    """Return derivative of logistic function."""
    e_pow_x = numpy.exp(x)
    try:
        # Replace inf's with 1, because with infinite precision,
        # output will be approximately 1
        # inf is caused by overflow in exp
        # NOTE: More efficient to replace non-infs
        out = numpy.ones(x.shape)
        not_infs = e_pow_x != numpy.Infinity
        out[not_infs] = e_pow_x[not_infs] / (e_pow_x[not_infs] + 1.0)**2
        return out
    except AttributeError:  # Scalar
        # Check for overflow
        if e_pow_x == INFINITY:
            # Return 1 if overflow, because with infinite precision,
            # output will be approximately 1
            return 1.0
        else:  # No overflow
            return e_pow_x / (e_pow_x + 1.0)**2


def tanh(x):
    """Sigmoid like function using tanh."""
    return numpy.tanh(x)


def dtanh(y):
    """Derivative of tanh."""
    return 1.0 - y**2


def gaussian(x, variance=1.0):
    return numpy.exp(-(x**2 / variance))


def dgaussian(x, y, variance=1.0):
    return -2.0 * x * y / variance


def relu(x):
    """Return ln(1 + e^x) for each input value."""
    # NOTE: numpy.errstate is very expensive, so the following is slower
    # Maybe numpy will optimize errstate in the future, to make this more effective
    # try:
    #     with numpy.errstate(over='raise'):
    #         return numpy.log(1.0 + numpy.exp(x))
    # except FloatingPointError:
    #     with numpy.errstate(over='ignore'):
    #         out = numpy.log(1.0 + numpy.exp(x))

    #     # Replace inf's with corresponding components in x
    #     # inf is caused by overflow in exp
    #     infs = out == numpy.Infinity
    #     out[infs] = x[infs]

    #     return out

    # Don't use try except with numpy.errstate, because it is slow
    out = numpy.log(1.0 + numpy.exp(x))

    # Replace inf's with corresponding components in x
    # inf is caused by overflow in exp
    infs = out == numpy.Infinity
    out[infs] = x[infs]

    return out


def drelu(x):
    """Return the derivative of the softplus relu function for x."""
    # NOTE: Can be optimized by caching numpy.e**(x) and returning e^x / (e^x + 1)
    return 1.0 / (1.0 + numpy.exp(-x))


def softmax(x):
    """Return the softmax of vector x."""
    # Subtract max to prevent overflow
    # Instead results in underflow for small components,
    # which is just zero, and thus acceptable
    # NOTE: Attempting to subtract max only when overflow would occur
    # (ex. try / except block for overflow with numpy.errstate('over': 'raise'))
    # results in worse performance for both the overflow and no overflow cases
    exp_ = numpy.exp(x - numpy.max(x, axis=-1, keepdims=True))
    return exp_ / numpy.sum(exp_, axis=-1, keepdims=True)


def dsoftmax(y):
    """Return the derivative of the softmax function for y."""
    # see http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
    # Compute matrix J, n x n, with y_i(1 - y_j) on the diagonals
    # and - y_i y_j on the non-diagonals
    # When getting errors multiply by error vector (J \vec{e})

    # Start with - y_i y_j matrix, then replace diagonal with y_i(1 - y_j)
    if len(y.shape) == 1:
        jacobian = numpy.outer(-y, y)
        jacobian[numpy.diag_indices(y.shape[0])] = y * (1 - y)
    elif len(y.shape) == 2:
        # Outer product each row of -y with each row of y.
        # using Einstein summation
        jacobian = numpy.einsum('ij...,i...->ij...', -y, y)
        iy, ix = numpy.diag_indices(y.shape[-1])
        jacobian[:, iy, ix] = y * (1. - y)
    else:
        raise ValueError('Unsupported tensor y in dsoftmax.')

    return jacobian
