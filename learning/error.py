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
    """Cross entropy error.

    Note that this error function is not symmetric.
    vec_a is expected to be the predicted vector,
    while vec_b is the reference vector.
    """
    def __call__(self, vec_a, vec_b):
        """Return the error between two vectors.

        Typically, vec_a with be a model output, and vec_b a target vector.
        """
        # TODO: Should we do something like numpy.maximum(vec_a, 1e-10) to prevent log(0)?
        # Use mean instead of sum, so magnitude is independent of length of vectors
        with numpy.errstate(divide='raise', invalid='raise'): # Do not allow log(-) or log(0)
            return -numpy.mean(numpy.log(vec_a) * vec_b)

    def derivative(self, vec_a, vec_b):
        """Return error, derivative_matrix."""
        # NOTE: If CE uses sum instead of mean, this would be -(vec_b / vec_a)
        return self(vec_a, vec_b), (vec_b / vec_a)/(-len(vec_b))
