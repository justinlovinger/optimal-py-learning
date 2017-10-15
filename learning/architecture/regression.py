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
"""Models for linear, logistic, and other forms of regression."""
import operator

import numpy

from learning import Model
from learning.optimize import Problem, BFGS, SteepestDescent
from learning.error import MSE

INITIAL_WEIGHTS_RANGE = 0.25


class RegressionModel(Model):
    """A model that optimizes the weight matrix of an equation of a set form.

    Args:
        attributes: int; Number of attributes in dataset.
        num_outputs: int; Number of output values in dataset.
            If onehot vector, this should equal the number of classes.
        optimizer: Instance of learning.optimize.optimizer.Optimizer.
        error_func: Instance of learning.error.ErrorFunc.
    """

    def __init__(self,
                 attributes,
                 num_outputs,
                 optimizer=None,
                 error_func=None):
        super(RegressionModel, self).__init__()

        # Weight matrix, optimized during training
        self._weight_matrix = self._random_weight_matrix((attributes,
                                                          num_outputs))

        # Optimizer to optimize weight_matrix
        if optimizer is None:
            # If there are a lot of weights, use an optimizer that doesn't use hessian
            # TODO (maybe): Default optimizer should work with mini-batches (be robust to changing problem)
            # optimizers like BFGS, and initial step strategies like FO and quadratic, rely heavily on information from
            # previous iterations, resulting in poor performance if the problem changes between iterations.
            # NOTE: Ideally, the Optimizer itself should handle its problem changing.

            # Count number of weights
            # NOTE: Cutoff value could use more testing
            if reduce(operator.mul, self._weight_matrix.shape) > 2500:
                # Too many weights, don't use hessian
                optimizer = SteepestDescent()
            else:
                # Low enough weights, use hessian
                optimizer = BFGS()

        self._optimizer = optimizer

        # Error function for training
        if error_func is None:
            error_func = MSE()
        self._error_func = error_func

    def reset(self):
        """Reset this model."""
        # Reset the weight matrix
        self._weight_matrix = self._random_weight_matrix(
            self._weight_matrix.shape)

        # Reset the optimizer
        self._optimizer.reset()

    def _random_weight_matrix(self, shape):
        """Return a random weight matrix."""
        # TODO: Random weight matrix should be a function user can pass in
        return (2 * numpy.random.random(shape) - 1) * INITIAL_WEIGHTS_RANGE

    def activate(self, input_vec):
        """Return the model outputs for given inputs."""
        return self._equation_output(input_vec)

    # TODO: Refactor, most of these functions are shared between
    # RBF, Regression, and MLP (models using Optimizers)
    def train_step(self, input_matrix, target_matrix):
        """Adjust the model towards the targets for given inputs.

        Train on a mini-batch.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        error, flat_weights = self._optimizer.next(
            Problem(
                obj_func=
                lambda xk: self._get_obj(xk, input_matrix, target_matrix),
                obj_jac_func=
                lambda xk: self._get_obj_jac(xk, input_matrix, target_matrix)),
            self._weight_matrix.ravel())
        self._weight_matrix = flat_weights.reshape(self._weight_matrix.shape)

        return error

    def _get_obj(self, flat_weights, input_matrix, target_matrix):
        """Helper function for Optimizer."""
        self._weight_matrix = flat_weights.reshape(self._weight_matrix.shape)
        return numpy.mean([
            self._error_func(self.activate(inp_vec), tar_vec)
            for inp_vec, tar_vec in zip(input_matrix, target_matrix)
        ])

    def _get_obj_jac(self, flat_weights, input_matrix, target_matrix):
        """Helper function for Optimizer."""
        self._weight_matrix = flat_weights.reshape(self._weight_matrix.shape)
        error, jacobian = self._get_jacobian(input_matrix, target_matrix)
        return error, jacobian.ravel()

    def _get_jacobian(self, input_matrix, target_matrix):
        """Return jacobian and error for given dataset."""
        errors, jacobians = zip(*[
            self._get_sample_jacobian(input_vec, target_vec)
            for input_vec, target_vec in zip(input_matrix, target_matrix)
        ])
        return numpy.mean(errors), numpy.mean(jacobians, axis=0)

    def _get_sample_jacobian(self, input_vec, target_vec):
        """Return jacobian and error for given sample."""
        output_vec = self.activate(input_vec)
        error, error_jac = self._error_func.derivative(output_vec, target_vec)
        # Each column of equations derivative corresponds to an output,
        # each row corresponds to an input.
        # Multiplying corresponding components of each row by the error_jac,
        # gives us the derivative.
        # NOTE: For most regression models (such as linear regression),
        # outputs are independent, and therefore each column will be the same
        # making this multiplication equivalent to column_vec.dot(error_jac)
        jacobian = self._equation_derivative(input_vec) * error_jac
        assert jacobian.shape == self._weight_matrix.shape

        return error, jacobian

    def _equation_output(self, input_vec):
        """Return the output of this models equation."""
        raise NotImplementedError()

    def _equation_derivative(self, input_vec):
        """Return the jacobian of this models equation, with regard to the weight matrix."""
        raise NotImplementedError()


class LinearRegressionModel(RegressionModel):
    r"""Regression model with an equation of the form: f(\vec{x}) = W \vec{x}."""

    def _equation_output(self, input_vec):
        """Return the output of this models equation."""
        return numpy.dot(input_vec, self._weight_matrix)

    def _equation_derivative(self, input_vec):
        """Return the jacobian of this models equation, with regard to the weight matrix."""
        # Jocobian, with regard to each output, is just the input vector
        # Tile simply repeats the input vector, for each output column
        # TODO: This should technically return a matrix of shape (num_outputs, num_weights),
        # instead of (num_outputs, num_inputs), but this will have a lot of 0s, and not be very efficient.
        return numpy.tile(input_vec[:, None], (1, self._weight_matrix.shape[1]))
