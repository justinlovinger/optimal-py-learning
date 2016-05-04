import numpy

from pynn import network

class SetOutputLayer(network.Layer):
    def __init__(self, output):
        super(SetOutputLayer, self).__init__()

        self.output = numpy.array(output)

    def activate(self, inputs):
        return self.output

    def reset(self):
        pass

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        pass

    def update(self, inputs, outputs, errors):
        pass

class RememberPatternsLayer(network.Layer):
    """Returns the output for a given input."""
    def __init__(self):
        super(RememberPatternsLayer, self).__init__()

        self._inputs_output_dict = {}

    def pre_training(self, patterns):
        for (inputs, targets) in patterns:
            self._inputs_output_dict[tuple(inputs)] = numpy.array(targets)

    def activate(self, inputs):
        return numpy.array(self._inputs_output_dict[tuple(inputs)])

    def reset(self):
        self._inputs_output_dict = {}

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        pass

    def update(self, inputs, outputs, errors):
        pass

class SummationLayer(network.Layer):
    def activate(self, inputs):
        return numpy.sum(inputs, axis=1)

    def reset(self):
        pass

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        pass

    def update(self, inputs, outputs, errors):
        pass

def approx_equal(a, b, tol=0.001):
    """Check if two numbers are about the same.

    Useful to correct for floating point errors.
    """
    return abs(a - b) < tol