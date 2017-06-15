import random
import math

import pytest
import numpy

from learning import calculate
from learning.testing import helpers

def test_protvecdiv_no_zero():
    assert (calculate.protvecdiv(
        numpy.array([1.0, 2.0, 3.0]), numpy.array([2.0, 2.0, 2.0]))
            == numpy.array([0.5, 1.0, 1.5])).all()

def test_protvecdiv_zero_den():
    # Returns 0 for position with 0 denominator
    # Also for 0 / 0
    assert (calculate.protvecdiv(
        numpy.array([1.0, 2.0, 0.0]), numpy.array([2.0, 0.0, 0.0]))
            == numpy.array([0.5, 0.0, 0.0])).all()

#######################
# Transfers
#######################
#####################
# Tanh
#####################
def test_tanh():
    assert helpers.approx_equal(calculate.tanh(numpy.array([-1.0, 0.0, 0.5, 1.0])),
                                [-0.761594, 0.0, 0.462117, 0.761594])

def test_tanh_gradient():
    helpers.check_gradient(calculate.tanh, lambda x: calculate.dtanh(calculate.tanh(x)))


#####################
# Gaussian
#####################
def test_gaussian_transfer():
    assert helpers.approx_equal(calculate.gaussian(numpy.array([-1.0, 0.0, 0.5, 1.0])),
                                [0.367879, 1.0, 0.778801, 0.367879])
    assert helpers.approx_equal(calculate.gaussian(numpy.array([-1.0, 0.0, 0.5, 1.0]), variance=0.5),
                                [0.135335, 1.0, 0.606531, 0.135335])

def test_gaussian_gradient():
    helpers.check_gradient(calculate.gaussian,
                           lambda x: calculate.dgaussian(x, calculate.gaussian(x)))

#####################
# Softmax
#####################
def test_softmax_transfer():
    assert list(calculate.softmax(numpy.array([1.0, 1.0]))) == [0.5, 0.5]

    assert helpers.approx_equal(calculate.softmax(numpy.array([1.0, 0.0])),
                                [0.7310585, 0.2689414])

    softmax_out = calculate.softmax(numpy.array([1.0, -1.0]))
    assert softmax_out[0] > 0.5 and softmax_out[1] < 0.5
    assert sum(softmax_out) == 1.0


def test_softmax_jacobian():
    helpers.check_gradient(calculate.softmax, lambda x: calculate.dsoftmax(calculate.softmax(x)),
                           f_shape='jac')

##############
# ReLU
##############
def test_relu_transfer():
    assert helpers.approx_equal(calculate.relu(numpy.array([0, 1])),
                                [0.6931471805, 1.3132616875])
    assert helpers.approx_equal(calculate.relu(numpy.array([-1.5, 10])),
                                [0.201413, 10.00004539])


def test_relu_derivative():
    assert helpers.approx_equal(calculate.drelu(numpy.array([0, 1])),
                                [0.5, 0.73105857])
    assert helpers.approx_equal(calculate.drelu(numpy.array([-1.5, 10])),
                                [0.182426, 0.9999546])


def test_relu_gradient():
    helpers.check_gradient(calculate.relu, calculate.drelu)
