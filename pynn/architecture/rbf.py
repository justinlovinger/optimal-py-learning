"""Layers required for an rbf implementation."""

import numpy

from pynn import network
from pynn.architecture import transfer

def fast_contribution(diffs, variance):
    return math.exp(-(diffs.dot(diffs)/variance))

class GaussianOutput(network.Layer):
    # TODO: rename this class
    required_prev = (transfer.GaussianTransfer,)

    def __init__(self, inputs, outputs,  learn_rate=1.0, normalize=False):
        super(GaussianOutput, self).__init__()

        self.learn_rate = learn_rate
        self.normalize = normalize

        self._size = (inputs, outputs)

        # Build weights matrix
        self._weights = numpy.zeros(self._size)
        self.reset()

    def reset(self):
        self._weights = numpy.zeros(self._size)

    def activate(self, inputs):
        output = numpy.dot(inputs, self._weights)
        if self.normalize:
            return output / numpy.sum(inputs)
        else:
            return output

    def get_prev_errors(self, errors, outputs):
        # TODO: test that this is correct
        deltas = errors * outputs
        return numpy.dot(deltas, self._weights.T)

    def update(self, all_inputs, outputs, all_errors):
        assert len(all_inputs) == 1
        inputs = all_inputs[0]
        # TODO: errors = numpy.average(all_errors)
        errors = all_errors[0]

        # Inputs are generally contributions
        if self.normalize:
            inputs = inputs / numpy.sum(inputs)

        # [:,None] quickly transposes an array to a col vector
        changes = inputs[:,None] * errors
        self._weights += self.learn_rate*changes