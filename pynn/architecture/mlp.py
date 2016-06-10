import random
import copy

import numpy

from pynn import network
from pynn.architecture import transfer

class AddBias(network.ParallelLayer):
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

    def update(self, all_inputs, outputs, all_errors):
        assert len(all_inputs) == 1
        inputs = all_inputs[0]
        self.layer.update([numpy.hstack((inputs, [1]))], outputs, all_errors)

class Perceptron(network.Layer):
    def __init__(self, inputs, outputs, 
                 learn_rate=0.5, momentum_rate=0.1, initial_weights_range=0.25):
        super(Perceptron, self).__init__()

        self.learn_rate = learn_rate
        self.momentum_rate = momentum_rate
        self.initial_weights_range = initial_weights_range

        self._size = (inputs, outputs)

        # Build weights matrix
        self._weights = numpy.zeros(self._size)
        self.reset()

        # Initial momentum
        self._momentums = numpy.zeros(self._size)

    def reset(self):
        # Randomize weights, between -initial_weights_range and initial_weights_range
        # TODO: randomize with Gaussian distribution instead of uniform. Mean = 0, small variance.
        random_matrix = numpy.random.random(self._size)
        self._weights = (2*random_matrix-1)*self.initial_weights_range

    def activate(self, inputs):
        return numpy.dot(inputs, self._weights)

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        errors = self._avg_all_errors(all_errors, outputs.shape)
        return numpy.dot(errors, self._weights.T)

    def update(self, all_inputs, outputs, all_errors):
        assert len(all_inputs) == 1
        inputs = all_inputs[0]
        deltas = self._avg_all_errors(all_errors, outputs.shape)

        # Update, [:,None] quickly transposes an array to a col vector
        changes = inputs[:,None] * deltas
        self._weights += self.learn_rate*changes + self.momentum_rate*self._momentums

        # Save change as momentum for next backpropogate
        self._momentums = changes


class DropoutPerceptron(Perceptron):
    """Perceptron that drops neurons during training."""
    def __init__(self, inputs, outputs, 
                 learn_rate=0.5, momentum_rate=0.1, initial_weights_range=0.25,
                 active_probability=0.5):
        super(DropoutPerceptron, self).__init__(inputs, outputs, learn_rate,
                                                momentum_rate, initial_weights_range)

        self._active_probability = active_probability
        self._active_neurons = range(self._size[1])
        self._full_weights = copy.deepcopy(self._weights)

    def pre_iteration(self, patterns):
        # Disable active neurons based on probability
        # TODO: ensure at least one neuron is active
        self._active_neurons = _random_indexes(self._size[1],
                                               self._active_probability)

        # TODO: Inspect previous DropoutPerceptron layer,
        # and adjust shape of matrix to account for new incoming neurons
        # Note: pre_iteration is called in activation order, so incoming
        # layers will always adjust output before this layer adjusts input
        incoming_active_neurons = range(self._size[0]) # TODO

        # Create a new weight matrix using slices from active_neurons
        # Stupid numpy hack for row and column slice to make sense
        # Otherwise, numpy will select specific elements, instead of rows and columns
        self._weights = self._full_weights[numpy.array(incoming_active_neurons)[:, None],
                                           self._active_neurons]

    def post_iteration(self, patterns):
        # Combine newly trained weights with full weight matrix
        # Override old weights with new weights for neurons
        assert 0

    def post_training(self, patterns):
        # Active all neurons
        # Scale weights by dropout probability, as a form of 
        # normalization by "expected" weight.

        # NOTE: future training iterations will continue with unscaled weights
        self._weights = self._full_weights * self._active_probability

        # Not really necessary, but could avoid confusion or future bugs
        self._active_neurons = range(self._size[1]) 


class DropoutInputs(network.Layer):
    """Passes inputs along unchanged, but disables inputs during training."""

    def __init__(self, inputs, active_probability=0.8):
        super(DropoutInputs, self).__init__()

        self._num_inputs = inputs
        self._active_probability = active_probability
        self._active_neurons = range(self._num_inputs)

    def reset(self):
        self._active_neurons = range(self._num_inputs)

    def activate(self, inputs):
        return inputs[self._active_neurons]

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return self._avg_all_errors(all_errors, outputs.shape)

    def update(self, all_inputs, outputs, all_errors):
        pass

    def pre_iteration(self, patterns):
        # Disable random selection of inputs
        self._active_neurons = _random_indexes(self._num_inputs,
                                               self._active_probability)

    def post_training(self, patterns):
        # All active when not training
        self._active_neurons = range(self._num_inputs)

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
    return selected_indexes


################################################
# Transfer functions with perceptron error rule
################################################
# The perceptron learning rule states that its errors
# are multiplied by the derivative of the transfers output.
# The output learning for an rbf, by contrast, does not.
class TanhTransferPerceptron(transfer.TanhTransfer):
    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return super(TanhTransferPerceptron, self).get_prev_errors(all_inputs, all_errors, outputs) * transfer.dtanh(outputs)


class ReluTransferPerceptron(transfer.ReluTransfer):
    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return super(ReluTransferPerceptron, self).get_prev_errors(all_inputs, all_errors, outputs) * transfer.drelu(outputs)


class LogitTransfer(transfer.LogitTransfer):
    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return super(LogitTransfer, self).get_prev_errors(all_inputs, all_errors, outputs) * transfer.dlogit(outputs)


class GaussianTransfer(transfer.GaussianTransfer):
    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return super(GaussianTransfer, self).get_prev_errors(all_inputs, all_errors, outputs) * transfer.dgaussian_vec(outputs)
