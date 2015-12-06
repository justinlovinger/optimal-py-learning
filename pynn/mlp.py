import numpy

import network

def sigmoid(x):
    """Sigmoid like function using tanh"""
    return numpy.tanh(x)

def dsigmoid(y):
    """Derivative of sigmoid above"""
    return 1.0 - y**2

class LinearPerceptron(network.Layer):
    def __init__(self, inputs, outputs, bias=True, 
                 learn_rate=0.5, momentum_rate=0.1, initial_weights_range=0.25):
        super(LinearPerceptron, self).__init__()

        self.learn_rate = learn_rate
        self.momentum_rate = momentum_rate
        self.initial_weights_range = initial_weights_range

        self._bias = bias
        if self._bias:
            self._size = (inputs+1, outputs)
        else:
            self._size = (inputs, outputs)

        # Build weights matrix
        self._weights = numpy.zeros(self._size)
        self.reset()

        # Initial momentum
        self._momentums = numpy.zeros(self._size)

    def reset(self):
        # Randomize weights, between -0.25 and 0.25
        random_matrix = numpy.random.random(self._size)
        self._weights = (2*random_matrix-1)*self.initial_weights_range

    def activate(self, inputs):
        if self._bias:
            inputs = numpy.hstack((inputs, [1]))

        if len(inputs) != self._size[0]:
            raise ValueError('wrong number of inputs')

        return numpy.dot(inputs, self._weights)

    def get_deltas(self, errors, outputs):
        return errors * outputs

    def get_errors(self, deltas):
        errors = numpy.dot(deltas, self._weights.T)
        if self._bias:
            return errors[:-1]
        return errors

    def update(self, inputs, deltas):
        if self._bias:
            inputs = numpy.hstack((inputs, [1]))

        # Update, [:,None] quickly transposes an array to a col vector
        changes = inputs[:,None] * deltas
        self._weights += self.learn_rate*changes + self.momentum_rate*self._momentums

        # Save change as momentum for next backpropogate
        self._momentums = changes

class SigmoidPerceptron(LinearPerceptron):
    def activate(self, inputs):
        return sigmoid(super(SigmoidPerceptron, self).activate(inputs))

    def get_deltas(self, errors, outputs):
        return super(SigmoidPerceptron, self).get_deltas(errors, dsigmoid(outputs))

def activate_test():
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]
    layers = [
                SigmoidPerceptron(2, 2, True),
                SigmoidPerceptron(2, 1, False),
             ]
    n = network.Network(layers)
    print n.activate(pat[3][0])

def train_test():
    import time
    #numpy.random.seed(0)

    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # Create a network with two input, two hidden, and one output nodes
    layers = [
                SigmoidPerceptron(2, 2, True),
                SigmoidPerceptron(2, 1, False),
             ]
    n = network.Network(layers)

    # Train it with some patterns
    start = time.clock()
    #n.logging = False
    n.train(pat, 1000, 0.0)
    print time.clock() - start
    # test it
    n.test(pat)

if __name__ == '__main__':
    train_test()