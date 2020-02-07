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
import math

import pytest
import numpy

from learning import calculate

from learning.testing import helpers


def test_protvecdiv_no_zero():
    assert (calculate.protvecdiv(
        numpy.array([1.0, 2.0, 3.0]),
        numpy.array([2.0, 2.0, 2.0])) == numpy.array([0.5, 1.0, 1.5])).all()


def test_protvecdiv_zero_den():
    # Returns 0 for position with 0 denominator
    # Also for 0 / 0
    assert (calculate.protvecdiv(
        numpy.array([1.0, 2.0, 0.0]),
        numpy.array([2.0, 0.0, 0.0])) == numpy.array([0.5, 0.0, 0.0])).all()


#######################
# Transfers
#######################
#####################
# Logit
#####################
def test_logit():
    assert calculate.logit(0) == 0.5
    assert helpers.approx_equal(
        calculate.logit(numpy.array([-1.0, 0.0, 0.5, 1.0])), [
            0.26894142,
            0.5,
            0.6224593,
            0.73105857,
        ])


def test_big_logit():
    assert calculate.logit(-1000.) == 0.0
    assert calculate.logit(1000.) == 1.0

    assert calculate.logit(-1000000.) == 0.0
    assert calculate.logit(1000000.) == 1.0


def test_dlogit_vector():
    helpers.check_gradient(
        calculate.logit,
        calculate.dlogit,
        f_shape='lin')


def test_dlogit_matrix():
    tensor_shape = [random.randint(1, 10) for _ in range(2)]

    helpers.check_gradient(
        lambda X: calculate.logit(X),
        lambda X: calculate.dlogit(X),
        f_arg_tensor=numpy.random.random(tensor_shape),
        f_shape='lin')


def test_big_dlogit():
    assert calculate.dlogit(-1000.) == 0.0
    assert calculate.dlogit(1000.) == 1.0

    assert calculate.dlogit(-1000000.) == 0.0
    assert calculate.dlogit(1000000.) == 1.0

    assert list(calculate.dlogit(numpy.array([-1000., 0., 1000.]))) == [
        0., 0.25, 1.
    ]


#####################
# Tanh
#####################
def test_tanh():
    assert helpers.approx_equal(
        calculate.tanh(numpy.array([-1.0, 0.0, 0.5, 1.0])),
        [-0.761594, 0.0, 0.462117, 0.761594])


def test_dtanh_vector():
    helpers.check_gradient(
        calculate.tanh,
        lambda x: calculate.dtanh(calculate.tanh(x)),
        f_shape='lin')


def test_dtanh_matrix():
    tensor_shape = [random.randint(1, 10) for _ in range(2)]

    helpers.check_gradient(
        lambda X: calculate.tanh(X),
        lambda X: calculate.dtanh(calculate.tanh(X)),
        f_arg_tensor=numpy.random.random(tensor_shape),
        f_shape='lin')


#####################
# Gaussian
#####################
def test_gaussian_transfer():
    assert helpers.approx_equal(
        calculate.gaussian(numpy.array([-1.0, 0.0, 0.5, 1.0])),
        [0.367879, 1.0, 0.778801, 0.367879])
    assert helpers.approx_equal(
        calculate.gaussian(numpy.array([-1.0, 0.0, 0.5, 1.0]), variance=0.5),
        [0.135335, 1.0, 0.606531, 0.135335])


def test_dgaussian_vector():
    helpers.check_gradient(
        calculate.gaussian,
        lambda x: calculate.dgaussian(x, calculate.gaussian(x)),
        f_shape='lin')


def test_dgaussian_matrix():
    tensor_shape = [random.randint(1, 10) for _ in range(2)]

    helpers.check_gradient(
        lambda X: calculate.gaussian(X),
        lambda X: calculate.dgaussian(X, calculate.gaussian(X)),
        f_arg_tensor=numpy.random.random(tensor_shape),
        f_shape='lin')


#####################
# Softmax
#####################
def test_softmax_vector():
    assert list(calculate.softmax(numpy.array([1.0, 1.0]))) == [0.5, 0.5]

    assert helpers.approx_equal(
        calculate.softmax(numpy.array([1.0, 0.0])), [0.7310585, 0.2689414])

    softmax_out = calculate.softmax(numpy.array([1.0, -1.0]))
    assert softmax_out[0] > 0.5 and softmax_out[1] < 0.5
    assert helpers.approx_equal(sum(softmax_out), 1.0)

    shape = random.randint(2, 10)
    softmax_out = calculate.softmax(numpy.array(sorted(numpy.random.random(shape))))
    assert sorted(softmax_out) == list(softmax_out)
    assert helpers.approx_equal(sum(softmax_out), 1.0)


def test_softmax_matrix():
    assert helpers.approx_equal(
        calculate.softmax(numpy.array([[1.0, 1.0], [1.0, 0.0]])),
        [[0.5, 0.5], [0.7310585, 0.2689414]])

    assert helpers.approx_equal(
        calculate.softmax(numpy.array([[1.0, 0.0], [0.5, 0.5]])),
        [[0.7310585, 0.2689414], [0.5, 0.5]])

    shape = (random.randint(2, 10), random.randint(2, 10))
    softmax_out = calculate.softmax(numpy.sort(numpy.random.random(shape), axis=1))
    assert (numpy.sort(softmax_out, axis=1) == softmax_out).all()
    assert helpers.approx_equal(numpy.sum(softmax_out, axis=1), numpy.ones(shape[0]))


def test_softmax_large_input():
    """Softmax includes an exponential, which can cause overflows.

    Our softmax implementation should protect against overflow.
    """
    assert list(calculate.softmax(numpy.array([-1000.0, 1000.0]))) == [
        0.0, 1.0
    ]


def test_dsoftmax_vector():
    helpers.check_gradient(
        calculate.softmax,
        lambda x: calculate.dsoftmax(calculate.softmax(x)),
        f_shape='jac')


def test_dsoftmax_matrix():
    tensor_shape = [random.randint(2, 10) for _ in range(2)]

    helpers.check_gradient(
        lambda X: calculate.softmax(X),
        lambda X: calculate.dsoftmax(calculate.softmax(X)),
        f_arg_tensor=numpy.random.random(tensor_shape),
        f_shape='jac-stack')


##############
# ReLU
##############
def test_relu_transfer():
    assert helpers.approx_equal(
        calculate.relu(numpy.array([0, 1])), [0.6931471805, 1.3132616875])
    assert helpers.approx_equal(
        calculate.relu(numpy.array([-1.5, 10])), [0.201413, 10.00004539])


def test_big_relu():
    """Naive relu can overflow with large input values."""
    assert helpers.approx_equal(
        calculate.relu(numpy.array([0., 1000.])), [0.6931471805, 1000])


def test_drelu_simple():
    assert helpers.approx_equal(
        calculate.drelu(numpy.array([0, 1])), [0.5, 0.73105857])
    assert helpers.approx_equal(
        calculate.drelu(numpy.array([-1.5, 10])), [0.182426, 0.9999546])


def test_big_drelu_simple():
    """Naive relu can overflow with large input values."""
    assert helpers.approx_equal(
        calculate.drelu(numpy.array([0., 1000.])), [0.5, 1.0])


def test_drelu_vector():
    helpers.check_gradient(calculate.relu, calculate.drelu, f_shape='lin')


def test_drelu_matrix():
    tensor_shape = [random.randint(1, 10) for _ in range(2)]

    helpers.check_gradient(
        lambda X: calculate.relu(X),
        lambda X: calculate.drelu(X),
        f_arg_tensor=numpy.random.random(tensor_shape),
        f_shape='lin')


def test_big_drelu():
    helpers.check_gradient(
        calculate.relu,
        calculate.drelu,
        f_arg_tensor=numpy.array([0., 1000.]),
        f_shape='lin')
