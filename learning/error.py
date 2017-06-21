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

        log(0) in vec_a is changed to very negative value, to keep the spirit
        of cross entropy, while avoiding numerical errors.
        """
        # Use mean instead of sum, so magnitude is independent of length of vectors
        with numpy.errstate(invalid='raise', divide='ignore'): # Do not allow log(-)
            log_a = numpy.log(vec_a)
        log_a = numpy.nan_to_num(log_a) # Change -inf (from log(0)) to -1.79769313e+308
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

        return self(vec_a, vec_b), vec_b_div_vec_a/(-len(vec_b))
