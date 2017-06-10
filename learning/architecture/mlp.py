import random
import copy

import numpy

from learning import Model
from learning.architecture import transfer

INITIAL_WEIGHTS_RANGE = 0.25

class MLP(Model):
    """MultiLayer Perceptron

    Args:
        shape: Number of inputs, followed by number of outputs of each layer.
            Shape of each weight matrix is given by sequential pairs in shape.
        tranfers: Optional. List of transfer layers.
            Can be given as a single transfer layer to easily define output transfer.
            Defaults to ReLU hidden followed by linear output.
        learn_rate: Learning rate for backpropogation.
        momentum_rate: Momentum rate for backpropagation, value is relative to learn_rate.
    """
    def __init__(self, shape, transfers=None, learn_rate=0.5, momentum_rate=0.2):
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

        #self._layers = self._make_layers(shape, transfers, learn_rate, learn_rate*momentum_rate)

        self._weight_matrices = []
        self._setup_weight_matrices()
        self._transfers = transfers

        # Gradient descent variables
        self._step_length = learn_rate
        self._momentum_rate = momentum_rate
        self._prev_jacobians = None

        # Setup activation vectors
        # 1 for input, then 2 for each hidden and output (1 for transfer, 1 for perceptron))
        self._weight_inputs = [numpy.zeros(shape[0])]
        self._transfer_inputs = []
        for size in shape[1:]:
            self._weight_inputs.append(numpy.zeros(size))
            self._transfer_inputs.append(numpy.zeros(size))

        self.reset()

    def _setup_weight_matrices(self):
        # TODO: Add bias
        self._weight_matrices = []
        num_inputs = self._shape[0]
        for num_outputs in self._shape[1:]:
            self._weight_matrices.append(self._random_weight_matrix((num_inputs, num_outputs)))
            num_inputs = num_outputs

    def _random_weight_matrix(self, shape):
        """Return a random weight matrix."""
        return (2*numpy.random.random(shape) - 1)*INITIAL_WEIGHTS_RANGE

    def reset(self):
        """Reset this model."""
        self._setup_weight_matrices()
        self._prev_jacobians = None

    def activate(self, inputs):
        """Return the model outputs for given inputs."""
        inputs = numpy.array(inputs)
        self._weight_inputs[0][:] = inputs

        for i, (weight_matrix, transfer_func) in enumerate(zip(self._weight_matrices, self._transfers)):
            # Track all activations for learning, and layer inputs
            self._transfer_inputs[i][:] = numpy.dot(self._weight_inputs[i], weight_matrix)
            self._weight_inputs[i+1][:] = transfer_func(self._transfer_inputs[i])

        # Return activation of the only layer that feeds into output
        return self._weight_inputs[-1]

    def _train_increment(self, input_vec, target_vec):
        """Train on a single input, target pair.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        jacobians, error = self._get_jacobians(input_vec, target_vec)

        # Gradient descent
        # TODO: Line search for step length
        for i, jacobian in enumerate(jacobians):
            self._weight_matrices[i] -= self._step_length * jacobian

        # Momentum
        # TODO: Line search for step length (or save previous step length and re-use it?)
        if self._prev_jacobians is not None:
            for i, jacobian in enumerate(self._prev_jacobians):
                self._weight_matrices[i] -= (self._step_length * self._momentum_rate) * jacobian
        self._prev_jacobians = jacobians


        return error

    def _get_jacobians(self, input_vec, target_vec):
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
            errors.append((errors[-1].dot(weight_matrix.T)
                           * transfer_func.derivative(
                               self._transfer_inputs[i], self._weight_inputs[i+1])))
        errors = reversed(errors)

        # Calculate jacobian for each weight matrix
        jacobians = []
        for i, error in enumerate(errors):
            jacobians.append(self._weight_inputs[i][:, None].dot(error[None, :]))

        return jacobians, output_error

class DropoutMLP(MLP):
    # TODO: Make sure the pre_iteration callback to disable neurons is implemented
    def __init__(self, shape, transfers=None, learn_rate=0.5, momentum_rate=0.1,
                 input_active_probability=0.8, hidden_active_probability=0.5):
        self._inp_act_prob = input_active_probability
        self._hid_act_prob = hidden_active_probability
        super(DropoutMLP, self).__init__(shape, transfers, learn_rate, momentum_rate)

        # Prepend new activations for dropout inputs layer
        self._activations.insert(0, numpy.zeros(shape[0]))

    def _make_layers(self, shape, transfers, learn_rate, momentum_rate):
        """Return sequence of perceptron and transfer layers."""
        # First layer is a special layer that disables inputs during training
        # Next is a special perceptron layer with bias (bias added by DropoutInputs)
        # NOTE: DropoutInputs also acts as bias layer
        dropout_inputs_layer = DropoutInputs(shape[0], self._inp_act_prob)
        biased_perceptron = DropoutPerceptron(shape[0]+1, shape[1], dropout_inputs_layer,
                                              learn_rate, momentum_rate,
                                              active_probability=self._hid_act_prob)
        layers = [dropout_inputs_layer,
                  biased_perceptron, transfers[0]]

        # After are hidden layers with given shape
        num_inputs = shape[1]
        dropout_perceptron = biased_perceptron
        for num_outputs, transfer_layer in zip(shape[2:-1], transfers[1:-1]):
            dropout_perceptron = DropoutPerceptron(num_inputs, num_outputs, dropout_perceptron,
                                                   learn_rate, momentum_rate,
                                                   active_probability=self._hid_act_prob)

            # Add perceptron followed by transfer
            layers.append(dropout_perceptron)
            layers.append(transfer_layer)

            num_inputs = num_outputs

        # Last perceptron layer must not reduce number of outputs
        layers.append(DropoutPerceptron(shape[-2], shape[-1], dropout_perceptron,
                                        learn_rate, momentum_rate,
                                        active_probability=1.0))
        layers.append(transfers[-1])

        return layers

    def train(self, *args, **kwargs):
        super(DropoutMLP, self).train(*args, **kwargs)

        # Post training callbacks
        for layer in self._layers:
            try:
                layer.post_training(None, None)
            except AttributeError: # Not dropout layer
                # Ignore transfer layers
                pass

    def pre_iteration(self, input_matrix, target_matrix):
        for layer in self._layers:
            try:
                layer.pre_iteration(input_matrix, target_matrix)
            except AttributeError: # Not dropout layer
                # Ignore transfer layers
                pass

    def post_iteration(self, input_matrix, target_matrix):
        for layer in self._layers:
            try:
                layer.post_iteration(input_matrix, target_matrix)
            except AttributeError: # Not dropout layer
                # Ignore transfer layers
                pass

class Layer(object):
    """A layer of computation for a supervised learning network."""
    attributes = tuple([]) # Attributes for this layer

    requires_prev = tuple([]) # Attributes that are required in the previous layer
    requires_next = tuple([]) # Attributes that are required in the next layer

    def __init__(self, *args, **kwargs):
        self.network = None

    def reset(self):
        raise NotImplementedError()

    def activate(self, inputs):
        raise NotImplementedError()

    def _avg_all_errors(self, all_errors, expected_shape):
        # For efficiency, and because it is a common case
        if len(all_errors) == 1:
            return all_errors[0]

        # Avg all non None errors
        sum = numpy.zeros_like(all_errors[0])
        num_averaged = 0
        for errors in all_errors:
            if errors is not None and errors.shape == expected_shape:
                sum += errors
                num_averaged += 1

        if num_averaged == 0:
            # No errors in lsit
            return None
        else:
            return sum / num_averaged

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        raise NotImplementedError()

    def update(self, all_inputs, outputs, all_errors):
        raise NotImplementedError()

    def pre_training(self, input_matrix, target_matrix):
        """Called before each training run.

        Optional.
        """

    def post_training(self, input_matrix, target_matrix):
        """Called after each training run.

        Optional.
        """

    def pre_iteration(self, input_matrix, target_matrix):
        """Called before each training iteration.

        Optional.
        """

    def post_iteration(self, input_matrix, target_matrix):
        """Called after each training iteration.

        Optional.
        """

class ParallelLayer(Layer):
    """A composite of layers connected in parallel."""

class AddBias(ParallelLayer):
    def __init__(self, layer):
        self.layer = layer

    def reset(self):
        self.layer.reset()

    def activate(self, inputs):
        # Add an extra input, always set to 1
        return self.layer.activate(numpy.hstack((inputs, [1])))

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        # Clip the last delta, which was for bias input
        return self.layer.get_prev_errors(all_inputs, all_errors, outputs)[:-1]

    def update(self, inputs, outputs, all_errors):
        self.layer.update(numpy.hstack((inputs, [1])), outputs, all_errors)

class Perceptron(Layer):
    def __init__(self, inputs, outputs,
                 learn_rate=0.5, momentum_rate=0.1, initial_weights_range=0.25):
        super(Perceptron, self).__init__()

        self.learn_rate = learn_rate
        self.momentum_rate = momentum_rate
        self.initial_weights_range = initial_weights_range

        self._size = (inputs, outputs)

        # Build weights matrix
        self._weights = numpy.zeros(self._size) # Randomized in reset

        # Initial momentum
        self._momentums = numpy.zeros(self._size)

        self.reset()

    def reset(self):
        # Randomize weights, between -initial_weights_range and initial_weights_range
        # TODO: randomize with Gaussian distribution instead of uniform. Mean = 0, small variance.
        self._weights = (2*numpy.random.random(self._size) - 1)*self.initial_weights_range
        self._momentums = numpy.zeros(self._size)

    def activate(self, inputs):
        return numpy.dot(inputs, self._weights)

    def get_prev_errors(self, input_vec, error_vec, output_vec):
        return numpy.dot(error_vec, self._weights.T)

    def update(self, inputs, outputs, error_vec):
        deltas = error_vec

        # Update, [:,None] quickly transposes an array to a col vector
        changes = inputs[:, None] * deltas
        self._weights += self.learn_rate*changes

        if self._momentums is not None:
            self._weights += self.momentum_rate*self._momentums

        # Save change as momentum for next backpropogate
        self._momentums = changes


class DropoutPerceptron(Perceptron):
    """Perceptron that drops neurons during training."""
    def __init__(self, inputs, outputs, incoming_layer,
                 learn_rate=0.5, momentum_rate=0.1, initial_weights_range=0.25,
                 active_probability=0.5):
        super(DropoutPerceptron, self).__init__(inputs, outputs, learn_rate,
                                                momentum_rate, initial_weights_range)

        self._incoming_layer = incoming_layer # For incoming active
        self._active_probability = active_probability
        self._active_neurons = None
        self._full_weights = None
        self.reset()

    def reset(self):
        super(DropoutPerceptron, self).reset()
        self._active_neurons = range(self._size[1])
        self._full_weights = copy.deepcopy(self._weights)

    def pre_iteration(self, input_matrix, target_matrix):
        # Disable active neurons based on probability
        self._active_neurons = _random_indexes(self._size[1],
                                               self._active_probability)

        # Inspect previous DropoutPerceptron layer,
        # and adjust shape of matrix to account for new incoming neurons
        # Note: pre_iteration is called in activation order, so incoming
        # layers will always adjust output before this layer adjusts input
        incoming_active_neurons = self._get_incoming_active_neurons()

        # Create a new weight matrix using slices from active_neurons
        # Stupid numpy hack for row and column slice to make sense
        # Otherwise, numpy will select specific elements, instead of rows and columns
        self._weights = self._full_weights[numpy.array(incoming_active_neurons)[:, None],
                                           self._active_neurons]

        # Invalidate previous momentums, since weight matrix changed
        self._momentums = None

    def post_iteration(self, input_matrix, target_matrix):
        # Combine newly trained weights with full weight matrix
        # Override old weights with new weights for neurons
        incoming_active_neurons = self._get_incoming_active_neurons()

        # We use numpy broadcasting to set the active rows and columns to
        # the reduced weight matrix
        # Testing indicates that this gives the expected result
        self._full_weights[numpy.array(incoming_active_neurons)[:, None],
                           self._active_neurons] = self._weights

    def _get_incoming_active_neurons(self):
        """Return list of active neurons in a preceeding dropout layer."""
        return self._incoming_layer._active_neurons

    def post_training(self, input_matrix, target_matrix):
        # Active all neurons
        # Scale weights by dropout probability, as a form of
        # normalization by "expected" weight.

        # NOTE: future training iterations will continue with unscaled weights
        self._weights = self._full_weights * self._active_probability

        # Not really necessary, but could avoid confusion or future bugs
        self._active_neurons = range(self._size[1])


class DropoutInputs(Layer):
    """Passes inputs along unchanged, but disables inputs during training.

    Also adds bias, since getting DropoutPerceptron to work with AddBias
    somewhat difficult.
    """

    def __init__(self, inputs, active_probability=0.8):
        super(DropoutInputs, self).__init__()

        self._num_inputs = inputs
        self._active_probability = active_probability
        self._active_neurons = None # set in reset
        self.reset()

    def reset(self):
        self._active_neurons = range(self._num_inputs+1) # Add bias

    def activate(self, inputs):
        # Active inputs, ignoring always active bias
        active_inputs = inputs[self._active_neurons[:-1]]

        # Add bias
        return numpy.hstack((active_inputs, [1.0]))

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return self._avg_all_errors(all_errors, outputs.shape)

    def update(self, all_inputs, outputs, all_errors):
        pass

    def pre_iteration(self, input_matrix, target_matrix):
        # Disable random selection of inputs
        self._active_neurons = _random_indexes(self._num_inputs,
                                               self._active_probability)

        # Bias is always active
        self._active_neurons.append(self._num_inputs)

    def post_training(self, input_matrix, target_matrix):
        # All active when not training
        self._active_neurons = range(self._num_inputs)

        # Bias is always active
        self._active_neurons.append(self._num_inputs)

def _random_indexes(length, probability):
    """Returns a list of indexes randomly selected from 0..length-1.

    Args:
        length: int; Maximum number of indexes.
        probability: float; Independant probability for any index to be selected.

    Returns:
        list<int>; List of selected indexes.
    """
    selected_indexes = []
    for i in range(length):
        if random.uniform(0.0, 1.0) <= probability:
            selected_indexes.append(i)

    # Do not allow no indexes to be selected
    if selected_indexes == []:
        selected_indexes.append(random.randint(0, length-1))

    return selected_indexes


################################################
# Transfer functions with perceptron error rule
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

class LinearTransfer(Transfer):
    def __call__(self, input_vec):
        return input_vec

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        return input_vec


class TanhTransfer(Transfer):
    def __call__(self, input_vec):
        return transfer.tanh(input_vec)

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        return transfer.dtanh(output_vec)


class ReluTransfer(Transfer):
    """Smooth approximation of a rectified linear unit (ReLU).

    Also known as softplus.
    """
    def __call__(self, input_vec):
        return transfer.relu(input_vec)

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        return transfer.drelu(input_vec)


class LogitTransfer(Transfer):
    pass


class GaussianTransfer(Transfer):
    def __init__(self, variance=1.0):
        super(GaussianTransfer, self).__init__()

        self._variance = variance

    def __call__(self, input_vec):
        return transfer.gaussian(input_vec, self._variance)

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        return transfer.dgaussian(input_vec, output_vec, self._variance)


class SoftmaxTransfer(Transfer):
    def __call__(self, input_vec):
        return transfer.softmax(input_vec)

    def derivative(self, input_vec, output_vec):
        """Return the derivative of this function.

        We take both input and output vector because
        some derivatives can be more efficiently calculated from
        the output of this function.
        """
        transfer.dsoftmax(output_vec)
