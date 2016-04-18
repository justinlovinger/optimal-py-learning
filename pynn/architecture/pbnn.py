import numpy

from pynn import network

class StoreInputsLayer(network.Layer):
    requires_prev = (None,)

    def __init__(self):
        self.stored_inputs = None

    def reset(self):
        self.stored_inputs = None

    def pre_training(self, patterns):
        # Extract inputs from patterns
        inputs = [p[0] for p in patterns]

        # And store them to recall later
        self.stored_inputs = numpy.array(inputs)

    def activate(self, inputs):
        # NOTE: should this copy the inputs before returning?
        # it would be less efficient, but less prone to error
        # Ideally, the recieving layer will make a copy if necessary
        return self.stored_inputs

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return None

    def update(self, inputs, outputs, errors):
        pass


class DistancesLayer(network.Layer):
    def activate(self, inputs, centers):
        diffs = inputs - centers
        distances = [numpy.sqrt(d.dot(d)) for d in diffs]
        return numpy.array(distances)

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        # TODO
        return None


class WeightedSummationLayer(network.Layer):
    def __init__(self):
        self.stored_targets = None

    def reset(self):
        self.stored_targets = None

    def pre_training(self, patterns):
        # Extract targets from patterns
        targets = [p[1] for p in patterns]

        # And store them to recall later
        self.stored_targets = numpy.array(targets)

    def activate(self, inputs):
        # Multiply each target (row of stored targets) by corresponding input element
        return numpy.sum(self.stored_targets * inputs[:, numpy.newaxis], axis=0)

    def update(self, inputs, outputs, errors):
        pass

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return None