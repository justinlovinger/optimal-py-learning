import numpy

from pynn import network

class SetOutputLayer(network.Layer):
    def __init__(self, output):
        super(SetOutputLayer, self).__init__()

        self.output = output

    def activate(self, inputs):
        return self.output

    def reset(self):
        pass

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        pass

    def update(self, inputs, outputs, errors):
        pass

class SetOutputPerInputsLayer(network.Layer):
    """Returns a set output for a defined input."""
    def __init__(self, inputs_output_dict):
        super(SetOutputPerInputsLayer, self).__init__()

        self._inputs_output_dict = inputs_output_dict

    def activate(self, inputs):
        return numpy.array(self._inputs_output_dict[tuple(inputs)])

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