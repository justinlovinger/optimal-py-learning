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

class MSE(ErrorFunc):
    """Mean squared error."""
    def __call__(self, vec_a, vec_b):
        """Return the error between two vectors.

        Typically, vec_a with be a model output, and vec_b a target vector.
        """
        return numpy.mean((vec_a - vec_b)**2)

    def derivative(self, vec_a, vec_b):
        """Return error, derivative_matrix."""
        error_vec = vec_a - vec_b
        mse = numpy.mean(error_vec**2) # For returning error

        # Note that error function is not 0.5*mse, so we multiply by 2
        error_vec *= (2.0/len(vec_b))

        return mse, error_vec

class CrossEntropy(ErrorFunc):
    """Cross entropy error."""
    # TODO
