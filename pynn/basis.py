import numpy
import math

import network

def distance(vec_a, vec_b):
    return numpy.linalg.norm(numpy.subtract(vec_a, vec_b))
    
def gaussian(x, variance=1.0):
    return math.exp(-(x**2/variance))
gaussian_vec = numpy.vectorize(gaussian)

def dgaussian(y, variance):
    return 2*y*gaussian(y, variance) / variance

def fast_contribution(diffs, variance):
    return math.exp(-(diffs.dot(diffs)/variance))

class GaussianLayer(network.Layer):
    """Equivalent to gaussian transfer, then perceptron."""
    def __init__(self, inputs, outputs, variance=1.0, learn_rate = 0.5):
        super(GaussianLayer, self).__init__()

        self.learn_rate = 0.5

        self._variance = variance
        self._weights = numpy.ones((inputs, outputs)) 
        self._contributions = numpy.ones(inputs)

    def reset(self):
        self._weights = numpy.ones(self._weights.shape) 

    def activate(self, inputs):
        self._contributions = gaussian_vec(inputs, self._variance)
        return numpy.dot(self._contributions, self._weights)

    def get_deltas(self, errors, outputs):
        return errors

    def get_errors(self, deltas, outputs):
        #errors = numpy.dot(deltas, self._weights.T)
        return dgaussian(deltas*outputs, self._variance)

    def update(self, inputs, deltas):
        # Update, [:,None] quickly transposes an array to a col vector
        changes = self._contributions[:,None] * deltas
        self._weights += self.learn_rate*changes

def train_test():
    import time
    #numpy.random.seed(0)
    import mlp

    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # Create a network with two input, two hidden, and one output nodes
    layers = [
                mlp.SigmoidPerceptron(2, 2),
                mlp.SigmoidPerceptron(2, 1),
                GaussianLayer(1, 1),
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