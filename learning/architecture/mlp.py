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

import random
import copy
import functools
import operator

import numpy

from learning import calculate, optimize
from learning import Model, LinearTransfer, ReluTransfer, MeanSquaredError
from learning.transfer import Transfer
from learning.optimize import Problem, SteepestDescent

INITIAL_WEIGHTS_RANGE = 0.25


# TODO: Add support for penalty functions
class MLP(Model):
    """MultiLayer Perceptron

    Args:
        shape: Number of inputs, followed by number of outputs of each layer.
            Shape of each weight matrix is given by sequential pairs in shape.
        transfers: Optional. List of transfer layers.
            Can be given as a single transfer layer to easily define output transfer.
            Defaults to ReLU hidden followed by linear output.
        optimizer: Optimizer; Optimizer used to optimize weight matrices.
        error_func: ErrorFunc; Error function for optimizing weight matrices.
    """

    def __init__(self, shape, transfers=None, optimizer=None, error_func=None):
        super(MLP, self).__init__()

        if transfers is None:
            transfers = [ReluTransfer() for _ in range((len(shape) - 2))
                         ] + [LinearTransfer()]
        elif isinstance(transfers, Transfer):
            # Treat single given transfer as output transfer
            transfers = [ReluTransfer()
                         for _ in range((len(shape) - 2))] + [transfers]

        if len(transfers) != len(shape) - 1:
            raise ValueError(
                'Must have exactly 1 transfer between each pair of layers, and after the output'
            )

        self._shape = shape

        self._bias_vec = self._random_weight_matrix(
            shape[1])  # Number of outputs of first layer
        self._weight_matrices = []
        self._setup_weight_matrices()
        self._transfers = transfers

        # Parameter optimization for training
        if optimizer is None:
            optimizer = optimize.make_optimizer(
                sum([
                    reduce(operator.mul, weight_matrix.shape)
                    for weight_matrix in self._weight_matrices
                ]))

        self._optimizer = optimizer

        # Error function for training
        if error_func is None:
            error_func = MeanSquaredError()
        self._error_func = error_func

        # Activation vectors
        # 1 for input, then 2 for each hidden and output (1 for transfer, 1 for perceptron))
        # To help with jacobian calculation
        self._weight_inputs = [None]*(len(self._shape))
        self._transfer_inputs = [None]*(len(self._shape)-1)

        self.reset()

    def _setup_weight_matrices(self):
        """Initialize weight matrices."""
        self._weight_matrices = []
        num_inputs = self._shape[0]
        for num_outputs in self._shape[1:]:
            self._weight_matrices.append(
                self._random_weight_matrix((num_inputs, num_outputs)))
            num_inputs = num_outputs

    def _random_weight_matrix(self, shape):
        """Return a random weight matrix."""
        # TODO: Random weight matrix should be a function user can pass in
        return (2 * numpy.random.random(shape) - 1) * INITIAL_WEIGHTS_RANGE

    def reset(self):
        """Reset this model."""
        self._setup_weight_matrices()
        self._optimizer.reset()

    def activate(self, input_vec):
        """Return the model outputs for given input_vec."""
        if len(input_vec) != self._shape[0]:
            raise ValueError('input_vec shape == %s, expected %s' %
                             (len(input_vec), self._shape[0]))

        self._weight_inputs[0] = input_vec
        # First part includes bias vector
        self._transfer_inputs[0] = numpy.dot(
            self._weight_inputs[0], self._weight_matrices[0]) + self._bias_vec
        self._weight_inputs[1] = self._transfers[0](self._transfer_inputs[0])

        for i, (weight_matrix, transfer_func) in list(
                enumerate(zip(self._weight_matrices, self._transfers)))[1:]:
            # Track all activations for learning, and layer inputs
            self._transfer_inputs[i] = numpy.dot(self._weight_inputs[i],
                                                 weight_matrix)
            self._weight_inputs[i + 1] = transfer_func(
                self._transfer_inputs[i])

        # Return activation of the only layer that feeds into output
        return numpy.copy(self._weight_inputs[-1])

    def train_step(self, input_matrix, target_matrix):
        """Adjust the model towards the targets for given inputs.

        Train on a mini-batch.
        """
        problem = Problem(
            obj_func=functools.partial(_mlp_obj, self, input_matrix,
                                       target_matrix),
            obj_jac_func=functools.partial(_mlp_obj_jac, self, input_matrix,
                                           target_matrix))

        error, flat_weights = self._optimizer.next(
            problem, _flatten(self._bias_vec, self._weight_matrices))
        self._bias_vec, self._weight_matrices = _unflatten_weights(
            flat_weights, self._shape)

        return error

    def _post_train(self):
        """Call after Model.train.

        Optional.
        """
        # Reset optimizer, because problem may change on next train call
        self._optimizer.reset()

    def _get_jacobians(self, input_matrix, target_matrix):
        """Return mean jacobian matrix for each weight matrix.

        Also return mean error.
        """
        # Calculate jacobian for each samples
        sample_bias_jacobians = []
        sample_weight_jacobians = []
        errors = []
        for input_vec, target_vec in zip(input_matrix, target_matrix):
            error, bias_jacobian, weight_jacobians = self._get_sample_jacobians(
                input_vec, target_vec)
            sample_bias_jacobians.append(bias_jacobian)
            sample_weight_jacobians.append(weight_jacobians)
            errors.append(error)

        # Average jacobians and error
        # NOTE: We don't use numpy.mean(sample_jacobians, axis=0) because it can raise errors
        return (numpy.mean(errors), numpy.mean(sample_bias_jacobians, axis=0),
                _mean_list_of_list_of_matrices(sample_weight_jacobians))

    def _get_sample_jacobians(self, input_vec, target_vec):
        """Return jacobian matrix for each weight matrix

        Also return error.
        """
        output_vec = self.activate(input_vec)

        error, error_jac = self._error_func.derivative(output_vec, target_vec)

        # TODO: Add optimization for cross entropy and softmax output (just o - t)
        # Derivative of error_vec w.r.t. output transfer
        error_jac = _dot_diag_or_matrix(error_jac,
                                        self._transfers[-1].derivative(
                                            self._transfer_inputs[-1],
                                            self._weight_inputs[-1]))

        # Calculate error for each row
        error_matrix = [error_jac]
        for i, (weight_matrix, transfer_func) in reversed(
                list(
                    enumerate(
                        zip(self._weight_matrices[1:], self._transfers[:
                                                                       -1])))):
            error_matrix.append(
                _dot_diag_or_matrix(
                    error_matrix[-1].dot(weight_matrix.T),
                    transfer_func.derivative(self._transfer_inputs[i],
                                             self._weight_inputs[i + 1])))
        error_matrix = list(reversed(error_matrix))

        # Calculate jacobian for each weight matrix
        jacobians = []
        for i, error_vec in enumerate(error_matrix):
            jacobians.append(
                self._weight_inputs[i][:, None].dot(error_vec[None, :]))

        return error, error_matrix[0], jacobians


def _dot_diag_or_matrix(vec, matrix):
    """Dot vector with either vector of diagonals or matrix.

    For efficiency, transfer derivatives can return either a vector corresponding
    to the diagonals of a jacobian, or a full jacobian.
    The diagonal must be multiplied element-wise, which is equivalent to
    a dot product with a diagonal matrix.
    """
    if len(matrix.shape) == 1:
        return vec * matrix
    else:
        return vec.dot(matrix)


def _mean_list_of_list_of_matrices(lol_matrices):
    """Return mean of each matrix in list of lists of matrices."""
    # Sum matrices
    mean_matrices = lol_matrices[0]
    for list_of_matrices in lol_matrices[1:]:
        for i, matrix in enumerate(list_of_matrices):
            mean_matrices[i] += matrix

    # Divide each by number of lists of matrices
    for matrix in mean_matrices:
        matrix /= len(lol_matrices)

    # Return list of mean matrices
    return mean_matrices


def _mlp_obj(model, input_matrix, target_matrix, parameters):
    model._bias_vec, model._weight_matrices = _unflatten_weights(
        parameters, model._shape)
    return numpy.mean([
        model._error_func(model.activate(inp_vec), tar_vec)
        for inp_vec, tar_vec in zip(input_matrix, target_matrix)
    ])


def _mlp_obj_jac(model, input_matrix, target_matrix, parameters):
    # TODO: Refactor so it doesn't need private attributes and methods
    model._bias_vec, model._weight_matrices = _unflatten_weights(
        parameters, model._shape)
    # Return error and flattened jacobians
    return (lambda obj, vec, jac: (obj, _flatten(vec, jac)))(
        *model._get_jacobians(input_matrix, target_matrix))


def _flatten(bias_vec, weight_matrices):
    """Flatten bias vector and weight matrices into flat vector."""
    return numpy.hstack([bias_vec] + [matrix.ravel() for matrix in weight_matrices])


def _unflatten_weights(vector, shape):
    """Unravel flat vector into bias vector and weight matrices."""
    bias_vec = vector[:shape[1]]
    matrices = []
    index = shape[1]
    for i, j in zip(shape[:-1], shape[1:]):
        matrices.append(vector[index:index + (i * j)].reshape((i, j)))
        index += (i * j)

    return bias_vec, matrices


class DropoutMLP(MLP):
    def __init__(self,
                 shape,
                 transfers=None,
                 optimizer=None,
                 error_func=None,
                 input_active_probability=0.8,
                 hidden_active_probability=0.5):
        if optimizer is None:
            # Don't use BFGS for Dropout
            # BFGS cannot effectively approximate hessian when problem
            # is constantly changing
            optimizer = SteepestDescent()

        super(DropoutMLP, self).__init__(shape, transfers, optimizer,
                                         error_func)

        # Dropout hyperparams
        self._inp_act_prob = input_active_probability
        self._hid_act_prob = hidden_active_probability

        # We modify transfers to disable hidden neurons
        # To disable inputs, we need a transfer for the input vector
        # To re-enable hidden neurons, we need to remember the original transfers
        self._input_transfer = LinearTransfer()
        self._real_transfers = self._transfers

        # We perform the post-training procedure on the first activation after training
        self._during_training = False
        self._did_post_training = True

    def activate(self, input_vec):
        """Return the model outputs for given inputs."""
        # Perform post-training procedure on the first activate after training.
        # If done during train method, post-training will not occur when model is used
        # incrementally
        if (not self._during_training) and (not self._did_post_training):
            self._post_training()
            self._did_post_training = True

        # Use input transfer to disable inputs (during training)
        return super(DropoutMLP,
                     self).activate(self._input_transfer(input_vec))

    def _post_training(self):
        # Activate all inputs
        self._input_transfer = LinearTransfer()

        # Activate all hidden
        self._transfers = self._real_transfers

        # Adjust weight matrices, based on active probabilities
        for i, _ in enumerate(self._weight_matrices):
            self._weight_matrices[i] *= self._hid_act_prob

    def train_step(self, input_matrix, target_matrix):
        """Adjust the model towards the targets for given inputs.

        Train on a mini-batch.
        """
        # Enter training mode
        self._during_training = True
        self._did_post_training = False

        # Disable inputs
        self._input_transfer = DropoutTransfer(
            LinearTransfer(), self._inp_act_prob, self._shape[0])

        # Disable hidden neurons
        self._disable_hiddens()

        error = super(DropoutMLP, self).train_step(input_matrix, target_matrix)

        # No longer in training mode
        self._during_training = False

        return error

    def _disable_hiddens(self):
        """Disable random neurons in hidden layers."""
        dropout_transfers = []

        # Don't disable output neurons
        for transfer_func, num_neurons in zip(self._real_transfers[:-1],
                                              self._shape[1:-1]):
            dropout_transfers.append(
                DropoutTransfer(transfer_func, self._hid_act_prob,
                                num_neurons))

        # Use original output transfer
        dropout_transfers.append(self._real_transfers[-1])

        self._transfers = dropout_transfers


################################################
# Transfer functions
################################################
class DropoutTransfer(Transfer):
    def __init__(self, transfer_func, active_probability, num_neurons):
        self._transfer = transfer_func
        self._active_neurons = _get_active_neurons(active_probability,
                                                   num_neurons)

    def __call__(self, input_vec):
        return self._transfer(input_vec) * self._active_neurons

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        return self._transfer.derivative(input_vec, output_vec)


def _get_active_neurons(active_probability, num_neurons):
    """Return list of active neurons."""
    if active_probability <= 0.0 or active_probability > 1.0:
        raise ValueError('0 < active_probability <= 1')

    active_neurons = [
        1.0 if random.uniform(0, 1) < active_probability else 0.0
        for _ in range(num_neurons)
    ]

    # Do not allow none active
    if 1.0 not in active_neurons:
        active_neurons[random.randint(0, len(active_neurons) - 1)] = 1.0

    return numpy.array(active_neurons)
