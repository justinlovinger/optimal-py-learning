﻿"""Transfer function layers."""

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
        return gaussian_vec(inputs, self._variance)

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return self._avg_all_errors(all_errors, outputs.shape)


class SoftmaxTransfer(Transfer):
    def activate(self, inputs):
        exp_ = numpy.exp(inputs)
        return exp_ / numpy.sum(exp_)


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
    return math.exp(-(x**2/variance))
gaussian_vec = numpy.vectorize(gaussian)

def dgaussian(y, variance):
    return 2*y*gaussian(y, variance) / variance
dgaussian_vec = numpy.vectorize(dgaussian)

def drelu(y):
    """Return the derivative of the softplus relu function for y."""
    return 1.0 / (1.0 + numpy.e**(-y))