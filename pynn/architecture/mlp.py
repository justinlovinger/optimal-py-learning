import numpy

from pynn import network

class AddBias(network.ParallelLayer):
    def __init__(self, layer):
        self.layer = layer
        self.num_inputs = self.layer.num_inputs - 1
        self.num_outputs = self.layer.num_outputs

    def reset(self):
        self.layer.reset()

    def activate(self, inputs):
        # Add an extra input, always set to 1
        return self.layer.activate(numpy.hstack((inputs, [1])))

    def get_prev_errors(self, errors, outputs):
        # Clip the last delta, which was for bias input
        return self.layer.get_prev_errors(errors, outputs)[:-1]

    def update(self, inputs, outputs, deltas):
        self.layer.update(numpy.hstack((inputs, [1])), outputs, deltas)

class Perceptron(network.Layer):
    def __init__(self, inputs, outputs, 
                 learn_rate=0.5, momentum_rate=0.1, initial_weights_range=0.25):
        super(Perceptron, self).__init__()
        self.num_inputs = inputs
        self.num_outputs = outputs

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

    def get_prev_errors(self, errors, outputs):
        return numpy.dot(errors, self._weights.T)

    def update(self, inputs, outputs, errors):
        deltas = errors * outputs

        # Update, [:,None] quickly transposes an array to a col vector
        changes = inputs[:,None] * deltas
        self._weights += self.learn_rate*changes + self.momentum_rate*self._momentums

        # Save change as momentum for next backpropogate
        self._momentums = changes