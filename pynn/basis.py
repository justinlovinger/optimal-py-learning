import numpy
import math

import network
    
def gaussian(x, variance=1.0):
    return math.exp(-(x**2/variance))
gaussian_vec = numpy.vectorize(gaussian)

def dgaussian(y, variance):
    return 2*y*gaussian(y, variance) / variance
dgaussian_vec = numpy.vectorize(dgaussian)

def fast_contribution(diffs, variance):
    return math.exp(-(diffs.dot(diffs)/variance))

class GaussianTransfer(network.Layer):
    def __init__(self, variance=1.0):
        super(GaussianTransfer, self).__init__()

        self._variance = variance

    def activate(self, inputs):
        return gaussian_vec(inputs, self._variance)

    def get_outputs(self, inputs, outputs):
        return dgaussian_vec(outputs, self._variance)

    def reset(self):
        pass

    def get_deltas(self, errors, outputs):
        return errors

    def get_errors(self, deltas, outputs):
        return deltas

    def update(self, inputs, deltas):
        pass

def train_test():
    import time
    #numpy.random.seed(9)
    import mlp

    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # Create a network with two input, two hidden, and one output nodes
    layers = [
                mlp.Perceptron(2, 2, True, learn_rate=0.3, momentum_rate=0.05),
                mlp.SigmoidTransfer(),
                mlp.Perceptron(2, 2, learn_rate=0.3, momentum_rate=0.05),
                mlp.SigmoidTransfer(),
                GaussianTransfer(),
                mlp.Perceptron(2, 1, learn_rate=0.3, momentum_rate=0.0),
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