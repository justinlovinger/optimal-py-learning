import numpy

from pynn import network

class Ensemble(network.ParallelLayer):
    def __init__(self, networks):
        super(Ensemble, self).__init__()

        self._networks = networks
        self.reset()

class Bagger(Ensemble):
    requires_prev = (None,)

    def reset(self):
        for network in self._networks:
            network.reset()

    def activate(self, inputs):
        # Unweighted average of layer outputs
        output = numpy.array(self._networks[0].activate(inputs), dtype='d')
        for network in self._networks[1:]:
            output += network.activate(inputs)

        return output / len(self._networks)

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return None #TODO

    def update(self, inputs, outputs, errors):
        for network in self._networks:
            network.update(inputs, errors + outputs)

