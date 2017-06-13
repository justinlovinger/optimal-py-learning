import random
import copy
import functools

import numpy

from learning import Model
from learning import calculate
from learning.optimize import Problem, SteepestDescent, SteepestDescentMomentum, SteepestDescentLineSearch

INITIAL_WEIGHTS_RANGE = 0.25

class MLP(Model):
    """MultiLayer Perceptron

    Args:
        shape: Number of inputs, followed by number of outputs of each layer.
            Shape of each weight matrix is given by sequential pairs in shape.
        transfers: Optional. List of transfer layers.
            Can be given as a single transfer layer to easily define output transfer.
            Defaults to ReLU hidden followed by linear output.
        make_optimizer_func: Function taking (model, input_matrix, target_matrix), and returning
            Optimizer object.
    """
    def __init__(self, shape, transfers=None, optimizer=None):
        super(MLP, self).__init__()

        if transfers is None:
            transfers = [ReluTransfer() for _ in range((len(shape)-2))]+[LinearTransfer()]
        elif isinstance(transfers, Transfer):
            # Treat single given transfer as output transfer
            transfers = [ReluTransfer() for _ in range((len(shape)-2))]+[transfers]

        if len(transfers) != len(shape)-1:
            raise ValueError(
                'Must have exactly 1 transfer between each pair of layers, and after the output')

        self._shape = shape

        self._weight_matrices = []
        self._setup_weight_matrices()
        self._transfers = transfers

        # Parameter optimization for training
        if optimizer is None:
            self._optimizer = SteepestDescentLineSearch()
        else:
            self._optimizer = optimizer

        # Setup activation vectors
        # 1 for input, then 2 for each hidden and output (1 for transfer, 1 for perceptron))
        # +1 for biases
        self._weight_inputs = [numpy.ones(shape[0]+1)]
        self._transfer_inputs = []
        for size in shape[1:]:
            self._weight_inputs.append(numpy.ones(size+1))
            self._transfer_inputs.append(numpy.zeros(size))

        self.reset()

    def _setup_weight_matrices(self):
        """Initialize weight matrices."""
        self._weight_matrices = []
        num_inputs = self._shape[0]
        for num_outputs in self._shape[1:]:
            # +1 for bias
            self._weight_matrices.append(self._random_weight_matrix((num_inputs+1, num_outputs)))
            num_inputs = num_outputs

    def _random_weight_matrix(self, shape):
        """Return a random weight matrix."""
        # TODO: Random weight matrix should be a function user can pass in
        return (2*numpy.random.random(shape) - 1)*INITIAL_WEIGHTS_RANGE

    def reset(self):
        """Reset this model."""
        self._setup_weight_matrices()
        self._optimizer.reset()

    def activate(self, inputs):
        """Return the model outputs for given inputs."""
        # [1:] because first component is bias
        self._weight_inputs[0][1:] = inputs

        for i, (weight_matrix, transfer_func) in enumerate(
                zip(self._weight_matrices, self._transfers)):
            # Track all activations for learning, and layer inputs
            self._transfer_inputs[i] = numpy.dot(self._weight_inputs[i], weight_matrix)
            # [1:] because first component is bias
            self._weight_inputs[i+1][1:] = transfer_func(self._transfer_inputs[i])

        # Return activation of the only layer that feeds into output
        # [1:] because first component is bias
        return self._weight_inputs[-1][1:]

    def train_step(self, input_matrix, target_matrix):
        """Adjust the model towards the targets for given inputs.

        Train on a mini-batch.
        """
        # TODO: Also pass obj_func (activation passed into error), for potentially more efficient line search
        problem = Problem(obj_jac_func=functools.partial(
            _mlp_obj_jac, self, input_matrix, target_matrix))

        error, flat_weights = self._optimizer.next(problem, _flatten(self._weight_matrices))
        self._weight_matrices = _unflatten_weights(flat_weights, self._shape)

        return error

    def _get_jacobians(self, input_matrix, target_matrix):
        """Return mean jacobian matrix for each weight matrix.

        Also return mean error.
        """
        # Calculate jacobian for each samples
        sample_jacobians = []
        errors = []
        for input_vec, target_vec in zip(input_matrix, target_matrix):
            jacobians, error = self._get_sample_jacobians(input_vec, target_vec)
            sample_jacobians.append(jacobians)
            errors.append(error)

        # Average jacobians and error
        return numpy.mean(errors), numpy.mean(sample_jacobians, axis=0)


    def _get_sample_jacobians(self, input_vec, target_vec):
        """Return jacobian matrix for each weight matrix

        Also return error.
        """
        # TODO: Should take error function
        outputs = self.activate(input_vec)

        error = outputs - target_vec
        output_error = numpy.mean(error**2) # For returning

        # Calculate error for each row
        errors = [error] # TODO: Use derivative of last layer, instead of assuming t - o
        for i, (weight_matrix, transfer_func) in reversed(
                list(enumerate(zip(self._weight_matrices[1:], self._transfers[:-1])))):
            # [1:] because first column corresponds to bias
            errors.append((errors[-1].dot(weight_matrix[1:].T)
                           * transfer_func.derivative(
                               # [1:] because first component is bias
                               self._transfer_inputs[i], self._weight_inputs[i+1][1:])))
        errors = reversed(errors)

        # Calculate jacobian for each weight matrix
        jacobians = []
        for i, error in enumerate(errors):
            jacobians.append(self._weight_inputs[i][:, None].dot(error[None, :]))

        return jacobians, output_error

def _mlp_obj_jac(model, input_matrix, target_matrix, parameters):
    # TODO: Refactor so it doesn't need private attributes and methods
    model._weight_matrices = _unflatten_weights(parameters, model._shape)
    # Return error and flattened jacobians
    return (lambda obj, jac: (obj, _flatten(jac)))(
        *model._get_jacobians(input_matrix, target_matrix))

def _flatten(matrices):
    return numpy.hstack([matrix.ravel() for matrix in matrices])

def _unflatten_weights(vector, shape):
    matrices = []
    index = 0
    for i, j in zip(shape[:-1], shape[1:]):
        i += 1 # For bias
        matrices.append(
            vector[index:index+(i*j)].reshape((i, j))
        )
        index += (i*j)

    return matrices

class DropoutMLP(MLP):
    def __init__(self, shape, transfers=None, optimizer=None,
                 input_active_probability=0.8, hidden_active_probability=0.5):
        super(DropoutMLP, self).__init__(shape, transfers, optimizer)

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
        return super(DropoutMLP, self).activate(self._input_transfer(input_vec))

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
        self._input_transfer = DropoutTransfer(LinearTransfer(), self._inp_act_prob, self._shape[0])

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
        for transfer_func, num_neurons in zip(self._real_transfers[:-1], self._shape[1:-1]):
            dropout_transfers.append(
                DropoutTransfer(transfer_func, self._hid_act_prob, num_neurons)
            )

        # Use original output transfer
        dropout_transfers.append(self._real_transfers[-1])

        self._transfers = dropout_transfers

################################################
# Transfer functions
################################################
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

class DropoutTransfer(Transfer):
    def __init__(self, transfer_func, active_probability, num_neurons):
        self._transfer = transfer_func
        self._active_neurons = _get_active_neurons(active_probability, num_neurons)

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

    active_neurons = [1.0 if random.uniform(0, 1) < active_probability else 0.0
                      for _ in range(num_neurons)]

    # Do not allow none active
    if 1.0 not in active_neurons:
        active_neurons[random.randint(0, len(active_neurons)-1)] = 1.0

    return numpy.array(active_neurons)

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


class LogitTransfer(Transfer):
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
        calculate.dsoftmax(output_vec)
