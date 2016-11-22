"""Transfer function layers."""

import numpy
import math

from pynn import network

class Transfer(network.Layer):
    def reset(self):
        pass

    def update(self, inputs, outputs, errors):
        pass

class TanhTransfer(Transfer):
    def activate(self, inputs):
        return tanh(inputs)

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return self._avg_all_errors(all_errors, outputs.shape)


class ReluTransfer(Transfer):
    """Smooth approximation of a rectified linear unit (ReLU).

    Also known as softplus.
    """
    
    def activate(self, inputs):
        """Return ln(1 + e^x) for each input value."""
        return numpy.log(1 + numpy.e**inputs)

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return self._avg_all_errors(all_errors, outputs.shape)


class LogitTransfer(Transfer):
    pass


class GaussianTransfer(Transfer):
    def __init__(self, variance=1.0):
        super(GaussianTransfer, self).__init__()

        self._variance = variance

    def activate(self, inputs):
        return gaussian(inputs, self._variance)

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return self._avg_all_errors(all_errors, outputs.shape)


class SoftmaxTransfer(Transfer):
    def __init__(self):
        super(SoftmaxTransfer, self).__init__()

        self._sum = None

    def activate(self, inputs):
        exp_ = numpy.exp(inputs)
        self._sum = numpy.sum(exp_)
        return exp_ / self._sum

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return self._avg_all_errors(all_errors, outputs.shape)

class NormalizeTransfer(Transfer):
    def activate(self, inputs, scaling_inputs):
        return inputs / numpy.sum(scaling_inputs)

    # TODO: what to do with get_prev_errors?
    # Look at current gaussian output for inspiration
    # divide errors by sum scaling inputs?
    def get_prev_errors(self, all_inputs, all_errors, outputs):
        # errors / sum(scaling_inputs)
        return (self._avg_all_errors(all_errors, outputs.shape) /
                numpy.sum(all_inputs[1]))
    
def tanh(x):
    """Sigmoid like function using tanh"""
    return numpy.tanh(x)

def dtanh(y):
    """Derivative of sigmoid above"""
    return 1.0 - y**2

def gaussian(x, variance=1.0):
    return numpy.exp(-(x**2/variance))

def dgaussian(y, variance):
    return 2*y*gaussian(y, variance) / variance

def drelu(y):
    """Return the derivative of the softplus relu function for y."""
    return 1.0 / (1.0 + numpy.e**(-y))

def dsoftmax(y):
    """Return the derivative of the softmax function for y."""
    # TODO: see http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
    # Compute matrix J, n x n, with y_i(1 - h_j) on the diagonals
    # and - y_i y_j on the non-diagonals
    # When getting erros multiply by error vector (J \vec{e})
    assert 0
