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

import numpy


class ErrorFunc(object):
    """An error function."""

    def __call__(self, vec_a, vec_b):
        """Return the error between two vectors.

        Typically, vec_a with be a model output, and vec_b a target vector.
        """
        raise NotImplementedError()

    def derivative(self, vec_a, vec_b):
        """Return (error, derivative matrix or vector)."""
        raise NotImplementedError()


class MeanSquaredError(ErrorFunc):
    """Mean squared error (MSE), defined by mean((vec_a - vec_b)^2)."""

    def __call__(self, vec_a, vec_b):
        """Return the error between two vectors.

        Typically, vec_a with be a model output, and vec_b a target vector.
        """
        return numpy.mean((numpy.subtract(vec_a, vec_b))**2)

    def derivative(self, vec_a, vec_b):
        """Return error, derivative_matrix."""
        error_vec = numpy.subtract(vec_a, vec_b)
        mse = numpy.mean(error_vec**2)  # For returning error

        # Note that error function is not 0.5*mse, so we multiply by 2
        error_vec *= (2.0 / len(vec_b))

        return mse, error_vec


class CrossEntropyError(ErrorFunc):
    """Cross entropy error, defined by -mean(log(vec_a) * vec_b).

    Note that this error function is not symmetric.
    vec_a is expected to be the predicted vector,
    while vec_b is the reference vector.
    """

    def __call__(self, vec_a, vec_b):
        """Return the error between two vectors.

        Typically, vec_a with be a model output, and vec_b a target vector.

        log(0) in vec_a is changed to very negative value, to keep the spirit
        of cross entropy, while avoiding numerical errors.
        """
        # Use mean instead of sum, so magnitude is independent of length of vectors
        with numpy.errstate(
                invalid='raise', divide='ignore'):  # Do not allow log(-)
            log_a = numpy.log(vec_a)
        # Change -inf (from log(0)) to -1.79769313e+308
        log_a = numpy.nan_to_num(log_a)
        return -numpy.mean(log_a * vec_b)

    def derivative(self, vec_a, vec_b):
        """Return error, derivative_matrix."""
        # NOTE: If CE uses sum instead of mean, this would be -(vec_b / vec_a)

        # Ignore 0/0 (handled in next line), warn for (x/0),
        # because this is less likely in practice, and may indicate a problem
        with numpy.errstate(invalid='ignore', divide='warn'):
            vec_b_div_vec_a = vec_b / vec_a

        # Change nan (0/0) to 0, and inf (x/0) to 1.79769313e+308
        vec_b_div_vec_a = numpy.nan_to_num(vec_b_div_vec_a)

        return self(vec_a, vec_b), vec_b_div_vec_a / (-len(vec_b))


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
