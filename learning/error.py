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
"""Error functions for use with some models."""

import operator

import numpy


class ErrorFunc(object):
    """An error function."""

    def __call__(self, tensor_a, tensor_b):
        """Return the error between two tensors.

        Typically, tensor_a is a model output, and tensor_b is a target tensor.
        """
        raise NotImplementedError()

    def derivative(self, tensor_a, tensor_b):
        """Return (error, derivative tensor)."""
        raise NotImplementedError()


class MeanSquaredError(ErrorFunc):
    """Mean squared error (MSE), defined by mean((tensor_a - tensor_b)^2)."""

    def __call__(self, tensor_a, tensor_b):
        """Return the error between two tensors.

        Typically, tensor_a is a model output, and tensor_b is a target tensor.
        """
        return numpy.mean((numpy.subtract(tensor_a, tensor_b))**2)

    def derivative(self, tensor_a, tensor_b):
        """Return (error, derivative tensor)."""
        error_vec = numpy.subtract(tensor_a, tensor_b)
        mse = numpy.mean(error_vec**2)  # For returning error

        # Note that error function is not 0.5*mse, so we multiply by 2
        error_vec *= (2.0 / reduce(operator.mul, tensor_a.shape))

        return mse, error_vec


class CrossEntropyError(ErrorFunc):
    """Cross entropy error, defined by -mean(log(tensor_a) * tensor_b).

    Note that this error function is not symmetric.
    tensor_a is expected to be the predicted tensor,
    while tensor_b is the reference tensor.

    Cross entropy error is typically used for classification problems,
    where tensor_b is a one-hot vector, or matrix of one-hot vectors.
    Likewise, tensor_a should be constrained to [0, 1], otherwise
    an error may be thrown (negative element) or error may be < 0 (> 1 element).
    """

    def __call__(self, tensor_a, tensor_b):
        """Return the error between two tensors.

        Typically, tensor_a is a model output, and tensor_b is a target tensor.

        log(0) in tensor_a is changed to very negative value, to keep the spirit
        of cross entropy, while avoiding numerical errors.
        """
        # Log tensor_a, allowing for log(0)
        with numpy.errstate(
                invalid='raise', divide='ignore'):  # Do not allow log(-)
            log_a = numpy.log(tensor_a)
        # Change -inf (from log(0)) to -1.79769313e+308
        log_a = numpy.nan_to_num(log_a)

        # If matrix, sum elements corresponding to each pattern,
        # and mean sums corresponding to patterns.
        # If vector, just sum
        return -numpy.mean(numpy.sum(log_a * tensor_b, axis=-1))

    def derivative(self, tensor_a, tensor_b):
        """Return (error, derivative tensor)."""
        # Ignore 0/0 (handled in next line), warn for (x/0),
        # because this is less likely in practice, and may indicate a problem
        with numpy.errstate(invalid='ignore', divide='warn'):
            tensor_b_div_tensor_a = tensor_b / tensor_a

        # Change nan (0/0) to 0, and inf (x/0) to 1.79769313e+308
        tensor_b_div_tensor_a = numpy.nan_to_num(tensor_b_div_tensor_a)

        error = self(tensor_a, tensor_b)
        if len(tensor_a.shape) == 1:  # Vector
            return error, -tensor_b_div_tensor_a
        else:  # Matrix or tensor
            return error, tensor_b_div_tensor_a / (
                -reduce(operator.mul, tensor_a.shape[:-1]))


#############################
# Penalty Functions
#############################
class PenaltyFunc(object):
    """A penalty function on weights."""
    derivative_uses_penalty = False

    def __init__(self, penalty_weight=1.0):
        super(PenaltyFunc, self).__init__()

        self._penalty_weight = penalty_weight

    def __call__(self, weight_tensor):
        """Return penalty of given weight tensor."""
        return self._penalty_weight * self._penalty(weight_tensor)

    def derivative(self, weight_tensor, penalty_output=None):
        """Return jacobian of given weight tensor.

        Output of this penalty function on the given weight_tensor
        can optionally be given for efficiently.
        Otherwise, it will be calculated if needed.
        """
        if self.derivative_uses_penalty:
            if penalty_output is None:
                penalty_output = self._penalty(weight_tensor)
            else:
                # Divide by self._penalty_weight,
                # because penalty_output is already multiplied by self._penalty_weight,
                # but we want to provide the raw penalty output
                penalty_output = penalty_output / self._penalty_weight

        return self._penalty_weight * self._derivative(weight_tensor,
                                                       penalty_output)

    def _penalty(self, weight_tensor):
        """Return penalty of given weight tensor."""
        raise NotImplementedError

    def _derivative(self, weight_tensor, penalty_output):
        """Return jacobian of given weight tensor."""
        raise NotImplementedError


# TODO: Test and document if these norms function the same for
# vectors and other tensors (because norms are defined differently for
# tensors, but we want vector norms)
class L1Penalty(PenaltyFunc):
    """Penalize weights by ||W||_1.

    Also known as Lasso.
    """
    def _penalty(self, weight_tensor):
        """Return penalty of given weight tensor."""
        return numpy.linalg.norm(weight_tensor, ord=1)

    def _derivative(self, weight_tensor, penalty_output):
        """Return jacobian of given weight tensor."""
        return numpy.sign(weight_tensor)


class L2Penalty(PenaltyFunc):
    """Penalize weights by ||W||_2.

    Also known as Lasso.
    """
    derivative_uses_penalty = True

    def _penalty(self, weight_tensor):
        """Return penalty of given weight tensor."""
        return numpy.linalg.norm(weight_tensor, ord=2)

    def _derivative(self, weight_tensor, penalty_output):
        """Return jacobian of given weight tensor."""
        return weight_tensor / penalty_output
