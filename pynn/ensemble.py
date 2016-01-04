from pynn import network

class Bagger(network.ParallelLayer):
    requires_prev = (None,)

    def reset(self):
        for network in self._networks:
            network.reset()

    def activate(self, inputs):
        # Unweighted average of layer outputs
        outputs = []
        for network in self._networks:
            outputs.append(network.activate(inputs))
        return sum(outputs) / len(outputs)

    def get_prev_errors(self, errors, outputs):
        return None #TODO

    def update(self, inputs, outputs, errors):
        for network in self._networks:
            network.update(inputs, errors + outputs)

