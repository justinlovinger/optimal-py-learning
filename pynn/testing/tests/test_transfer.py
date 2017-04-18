import random
import math

import pytest
import numpy
from scipy import optimize

from pynn.architecture import transfer
from pynn.testing import helpers

#####################
# Tanh
#####################
def test_tanh_transfer():
    layer = transfer.TanhTransfer()
    expected = [-0.761594, 0.0, 0.462117, 0.761594]
    output = layer.activate(numpy.array([-1.0, 0.0, 0.5, 1.0]))
    output = [round(v, 6) for v in output]
    assert output == expected


def test_tanh_gradient():
    check_gradient(transfer.tanh, lambda x: transfer.dtanh(transfer.tanh(x)))


#####################
# Gaussian
#####################
def test_gaussian_transfer():
    layer = transfer.GaussianTransfer()
    expected = [0.367879, 1.0, 0.778801, 0.367879]
    output = layer.activate(numpy.array([-1.0, 0.0, 0.5, 1.0]))
    output = [round(v, 6) for v in output]
    assert output == expected

    layer = transfer.GaussianTransfer(variance=0.5)
    expected = [0.135335, 1.0, 0.606531, 0.135335]
    output = layer.activate(numpy.array([-1.0, 0.0, 0.5, 1.0]))
    output = [round(v, 6) for v in output]
    assert output == expected


def test_gaussian_gradient():
    check_gradient(transfer.gaussian,
                   lambda x: transfer.dgaussian(x, transfer.gaussian(x)))

#####################
# Softmax
#####################
def test_softmax_transfer():
    layer = transfer.SoftmaxTransfer()

    assert list(layer.activate(numpy.array([1.0, 1.0]))) == [0.5, 0.5]

    expecteds = [0.7310585, 0.2689414]
    outputs = list(layer.activate(numpy.array([1.0, 0.0])))
    for output, expected in zip(outputs, expecteds):
        assert helpers.approx_equal(output, expected)

    output_ = list(layer.activate(numpy.array([1.0, -1.0])))
    assert output_[0] > 0.5 and output_[1] < 0.5
    assert sum(output_) == 1.0


def test_softmax_jacobian():
    check_gradient(transfer.softmax, lambda x: transfer.dsoftmax(transfer.softmax(x)),
                   jacobian=True)

@pytest.mark.skip(reason='Not sure approximate gradient is correct')
def test_softmax_gradient():
    inputs = numpy.random.rand(random.randint(2, 10))
    errors = numpy.random.rand(inputs.shape[0])
    assert check_gradient(
       lambda x: errors * transfer.softmax(x),
       lambda x: errors.dot(transfer.dsoftmax(transfer.softmax(x))),
       inputs=inputs)
    check_gradient(transfer.softmax, lambda x: transfer.dsoftmax(transfer.softmax(x)))

##############
# ReLU
##############
def test_relu_transfer():
    layer = transfer.ReluTransfer()

    assert helpers.approx_equal(list(layer.activate(numpy.array([0, 1]))),
                                [0.6931471805, 1.3132616875])
    assert helpers.approx_equal(list(layer.activate(numpy.array([-1.5, 10]))),
                                [0.201413, 10.00004539])


def test_relu_derivative():
    assert helpers.approx_equal(list(transfer.drelu(numpy.array([0, 1]))),
                                [0.5, 0.73105857])
    assert helpers.approx_equal(list(transfer.drelu(numpy.array([-1.5, 10]))),
                                [0.182426, 0.9999546])


def test_relu_gradient():
    check_gradient(transfer.relu, transfer.drelu)


##############
# Helpers
##############
def test_check_gradient():
    check_gradient(lambda x: x**2, lambda x: 2*x)
    check_gradient(lambda x: numpy.sqrt(x), lambda x: 1.0 / (2*numpy.sqrt(x)))


def test_check_gradient_jacobian():
    check_gradient(lambda x: numpy.array([x[0]**2*x[1], 5*x[0]+math.sin(x[1])]),
                   lambda x: numpy.array([[2*x[0]*x[1], x[0]**2       ],
                                          [5.0,         math.cos(x[1])]]),
                   inputs=numpy.random.rand(2),
                   jacobian=True)


def check_gradient(f, df, inputs=None, epsilon=1e-6, jacobian=False):
    if inputs is None:
        inputs = numpy.random.rand(random.randint(2, 10))

    if jacobian:
        approx_func = _approximate_jacobian
    else:
        approx_func = _approximate_gradient

    assert numpy.mean(numpy.abs(
        df(inputs) - approx_func(f, inputs, epsilon))) <= epsilon


def _approximate_gradient(f, x, epsilon):
    return numpy.array([_approximate_ith(i, f, x, epsilon) for i in range(x.shape[0])])


def _approximate_ith(i, f, x, epsilon):
    x_plus_i = x.copy()
    x_plus_i[i] += epsilon
    x_minus_i = x.copy()
    x_minus_i[i] -= epsilon
    return ((f(x_plus_i) - f(x_minus_i)) / (2*epsilon))[i]

def _approximate_jacobian(f, x, epsilon):
    jacobian = numpy.zeros((x.shape[0], x.shape[0]))
    # Jocobian has inputs on cols and outputs on rows
    for j in range(x.shape[0]):
        for i in range(x.shape[0]):
            x_plus_i = x.copy()
            x_plus_i[i] += epsilon
            x_minus_i = x.copy()
            x_minus_i[i] -= epsilon
            jacobian[j,i] = (f(x_plus_i)[j] - f(x_minus_i)[j])/(2.0*epsilon)
    return jacobian
