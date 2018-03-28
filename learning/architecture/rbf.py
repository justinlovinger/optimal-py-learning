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
"""Radial Basis Function network."""
import operator

import numpy

from learning import calculate, optimize, Model, SOM, MeanSquaredError
from learning.optimize import Problem

INITIAL_WEIGHTS_RANGE = 0.25


# TODO: Add support for penalty functions
class RBF(Model):
    """Radial Basis Function network.

    Args:
        attributes: int; Number of attributes in dataset.
        num_clusters: Number of clusters in intermediate layer.
        num_outputs: int; Number of output values in dataset.
            If onehot vector, this should equal the number of classes.
        optimizer: Instance of learning.optimize.optimizer.Optimizer.
        error_func: Instance of learning.error.ErrorFunc.
    """

    def __init__(self,
                 attributes,
                 num_clusters,
                 num_outputs,
                 optimizer=None,
                 error_func=None,
                 variance=None,
                 scale_by_similarity=True,
                 pre_train_clusters=False,
                 move_rate=0.1,
                 neighborhood=2,
                 neighbor_move_rate=1.0):
        super(RBF, self).__init__()

        # Clustering algorithm
        self._pre_train_clusters = pre_train_clusters
        self._som = SOM(
            attributes,
            num_clusters,
            move_rate=move_rate,
            neighborhood=neighborhood,
            neighbor_move_rate=neighbor_move_rate)

        # Variance for gaussian
        if variance is None:
            variance = 4.0 / num_clusters
        self._variance = variance

        # Weight matrix for output
        self._weight_matrix = self._random_weight_matrix((num_clusters,
                                                          num_outputs))

        # Optimizer to optimize weight_matrix
        if optimizer is None:
            optimizer = optimize.make_optimizer(
                reduce(operator.mul, self._weight_matrix.shape))

        self._optimizer = optimizer

        # Error function for training
        if error_func is None:
            error_func = MeanSquaredError()
        self._error_func = error_func

        # Optional scaling output by total gaussian similarity
        self._scale_by_similarity = scale_by_similarity

        # For training
        self._similarities = None
        self._total_similarity = None

    def reset(self):
        """Reset this model."""
        self._som.reset()
        self._optimizer.reset()

        self._weight_matrix = self._random_weight_matrix(
            self._weight_matrix.shape)

        self._similarities = None
        self._total_similarity = None

    def _random_weight_matrix(self, shape):
        """Return a random weight matrix."""
        # TODO: Random weight matrix should be a function user can pass in
        return (2 * numpy.random.random(shape) - 1) * INITIAL_WEIGHTS_RANGE

    def activate(self, inputs):
        """Return the model outputs for given inputs."""
        # Get distance to each cluster center, and apply gaussian for similarity
        self._similarities = calculate.gaussian(
            self._som.activate(inputs), self._variance)

        # Get output by weighted summation of similarities, weighted by weights
        output = numpy.dot(self._similarities, self._weight_matrix)

        if self._scale_by_similarity:
            self._total_similarity = numpy.sum(self._similarities)
            output /= self._total_similarity

        return output

    def train_step(self, input_matrix, target_matrix):
        """Adjust the model towards the targets for given inputs.

        Train on a mini-batch.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        # Train RBF
        error, flat_weights = self._optimizer.next(
            Problem(
                obj_func=
                lambda xk: self._get_obj(xk, input_matrix, target_matrix),
                obj_jac_func=
                lambda xk: self._get_obj_jac(xk, input_matrix, target_matrix)),
            self._weight_matrix.ravel())
        self._weight_matrix = flat_weights.reshape(self._weight_matrix.shape)

        # Train SOM clusters
        self._som.train_step(input_matrix, target_matrix)

        return error

    def _pre_train(self, input_matrix, target_matrix):
        """Call before Model.train.

        Optional.
        """
        if self._pre_train_clusters:
            # Train SOM first
            # TODO: RBF init should include dict for som train arguments
            self._som.train(input_matrix, target_matrix)

    def _post_train(self, input_matrix, target_matrix):
        """Call after Model.train.

        Optional.
        """
        # Reset optimizer, because problem may change on next train call
        self._optimizer.reset()

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

        if self._scale_by_similarity:
            error_jac /= self._total_similarity

        jacobian = self._similarities[:, None].dot(error_jac[None, :])

        return error, jacobian
