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
                 error_func=None,
                 penalty_func=None):
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

        # Penalty function for training
        self._penalty_func = penalty_func

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
        # Use an Optimizer to move weights in a direction that minimizes
        # error (as defined by given error function).
        error, flat_weights = self._optimizer.next(
            Problem(
                obj_func=
                lambda xk: self._get_obj(xk, input_matrix, target_matrix),
                obj_jac_func=
                lambda xk: self._get_obj_jac(xk, input_matrix, target_matrix)),
            self._weight_matrix.ravel())
        self._weight_matrix = flat_weights.reshape(self._weight_matrix.shape)

        return error

    ######################################
    # Helper functions for optimizer
    ######################################
    def _get_obj(self, flat_weights, input_matrix, target_matrix):
        """Helper function for Optimizer."""
        self._weight_matrix = flat_weights.reshape(self._weight_matrix.shape)
        return self._get_objective_value(input_matrix, target_matrix)

    def _get_objective_value(self, input_matrix, target_matrix):
        """Return error on given dataset."""
        error = numpy.mean([
            self._error_func(self.activate(inp_vec), tar_vec)
            for inp_vec, tar_vec in zip(input_matrix, target_matrix)
        ])

        if self._penalty_func is not None:
            # Error is mean of sample errors + weight penalty
            # NOTE: We ravel the weight matrix, to take vector norm
            error += self._penalty_func(self._weight_matrix.ravel())

        return error

    ######################################
    # Differentiation
    ######################################
    # for numerical optimization
    def _get_obj_jac(self, flat_weights, input_matrix, target_matrix):
        """Helper function for Optimizer."""
        self._weight_matrix = flat_weights.reshape(self._weight_matrix.shape)
        error, jacobian = self._get_jacobian(input_matrix, target_matrix)
        return error, jacobian.ravel()

    def _get_jacobian(self, input_matrix, target_matrix):
        """Return jacobian and error for given dataset."""
        # Calculate jacobian, given error function
        errors, jacobians = zip(*[
            self._get_sample_jacobian(input_vec, target_vec)
            for input_vec, target_vec in zip(input_matrix, target_matrix)
        ])
        error = numpy.mean(errors)
        jacobian = numpy.mean(jacobians, axis=0)


        # Calculate weight penalty
        if self._penalty_func is not None:
            # NOTE: We ravel the weight matrix, to take vector norm
            flat_weights = self._weight_matrix.ravel()
            penalty = self._penalty_func(flat_weights)
            penalty_jac = self._penalty_func.derivative(
                flat_weights,
                penalty_output=penalty).reshape(self._weight_matrix.shape)

            # Error and jacobian is combination of error and weight penalty
            error += penalty
            jacobian += penalty_jac
        
        return error, jacobian

    def _get_sample_jacobian(self, input_vec, target_vec):
        """Return jacobian and error for given sample."""
        output_vec = self.activate(input_vec)
        if output_vec.shape != target_vec.shape:
            raise ValueError(
                'target_vec.shape does not match output_vec.shape')

        error, error_jac = self._error_func.derivative(output_vec, target_vec)
        jacobian = self._equation_derivative(input_vec, error_jac)
        assert reduce(operator.mul, jacobian.shape) == reduce(
            operator.mul, self._weight_matrix.shape)

        return error, jacobian

    def _equation_output(self, input_vec):
        """Return the output of this models equation."""
        raise NotImplementedError()

    def _equation_derivative(self, input_vec, error_jac):
        """Return the jacobian of this models equation corresponding to the given error.

        Derivative with regard to weights.
        """
        raise NotImplementedError()


class LinearRegressionModel(RegressionModel):
    r"""Regression model with an equation of the form: f(\vec{x}) = W \vec{x}."""

    def _equation_output(self, input_vec):
        """Return the output of this models equation."""
        return numpy.dot(input_vec, self._weight_matrix)

    def _equation_derivative(self, input_vec, error_jac):
        """Return the jacobian of this models equation corresponding to the given error.

        Derivative with regard to weights.
        """
        return input_vec[:, None].dot(error_jac[None, :])


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

        return self._penalty_weight * self._derivative(weight_tensor, penalty_output)

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
