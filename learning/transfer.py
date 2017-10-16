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
"""Transfer functions

Useful for MLP.
"""
import numpy

from learning import calculate


class Transfer(object):
    def __call__(self, input_vec):
        raise NotImplementedError()

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        raise NotImplementedError()


class LinearTransfer(Transfer):
    def __call__(self, input_vec):
        return input_vec

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        return numpy.ones(input_vec.shape)


class TanhTransfer(Transfer):
    def __call__(self, input_vec):
        return calculate.tanh(input_vec)

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        return calculate.dtanh(output_vec)


class ReluTransfer(Transfer):
    """Smooth approximation of a rectified linear unit (ReLU).

    Also known as softplus.
    """

    def __call__(self, input_vec):
        return calculate.relu(input_vec)

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        return calculate.drelu(input_vec)


# TODO
class _LogitTransfer(Transfer):
    pass


class GaussianTransfer(Transfer):
    def __init__(self, variance=1.0):
        super(GaussianTransfer, self).__init__()

        self._variance = variance

    def __call__(self, input_vec):
        return calculate.gaussian(input_vec, self._variance)

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        return calculate.dgaussian(input_vec, output_vec, self._variance)


class SoftmaxTransfer(Transfer):
    def __call__(self, input_vec):
        return calculate.softmax(input_vec)

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        return calculate.dsoftmax(output_vec)
