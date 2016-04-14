import numpy

from pynn import network

class Ensemble(network.ParallelLayer):
    def __init__(self, networks):
        super(Ensemble, self).__init__()

        self.num_inputs = networks[0].num_inputs
        self.num_outputs = networks[0].num_outputs

        # TODO: validate that all networks have the same num inputs and outputs

        self._networks = networks
        self.reset()

class Bagger(Ensemble):
    requires_prev = (None,)

    def reset(self):
        for network in self._networks:
            network.reset()

    def activate(self, inputs):
        # Unweighted average of layer outputs
        output = numpy.zeros(self.num_outputs)
        for network in self._networks:
            output += network.activate(inputs)

        return output / len(self._networks)

    def get_prev_errors(self, errors, outputs):
        return None #TODO

    def update(self, inputs, outputs, errors):
        for network in self._networks:
            network.update(inputs, errors + outputs)

