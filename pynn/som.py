import numpy

from pynn import network
from pynn import transfer

def distance(vec_a, vec_b):
    diff = numpy.subtract(vec_a, vec_b)
    return diff.dot(diff)

class SOM(network.Layer):
    required_prev = (None,)

    def __init__(self, inputs, neurons):
        super(SOM, self).__init__()

        self._size = (neurons, inputs)
        self._weights = numpy.zeros(self._size)
        self._distances = numpy.zeros(self._size)
        self.reset()

    def reset(self):
        # Randomize weights, between -1 and 1
        random_matrix = numpy.random.random(self._size)
        self._weights = (2*random_matrix-1)*1.0

    def activate(self, inputs):
        diffs = inputs - self._weights
        self._distances = [d.dot(d) for d in diffs]
        return numpy.array(self._distances)

    def get_deltas(self, errors, outputs):
        return

    def get_errors(self, deltas, outputs):
        return

    def get_closest(self):
        closest = 0
        closest_d = self._distances[0]
        for i in range(1, len(self._distances)):
            d = self._distances[i]
            if d < closest_d:
                closest = i
                closest_d = d

        return closest

    def move_neurons(self, inputs, move_rate=0.1, 
                     neighborhood=2, neighbor_move_rate=1.0):
        # Perform a competition, and move the winner closer to the input
        closest = self.get_closest()

        # Move the winner and neighbors closer
        # The further the neighbor, the less it should move
        for i in range(closest-neighborhood, closest+neighborhood+1):
            if i >= 0 and i < self._size[0]: # if in range
                neighbor_distance = float(abs(i-closest))
                move_rate_modifier = transfer.gaussian(neighbor_distance, 
                                                       neighbor_move_rate)
                final_rate = move_rate_modifier*move_rate

                self._weights[i] += final_rate*(inputs-self._weights[i])

    def update(self, inputs, deltas):
        self.move_neurons(inputs)

def train_test():
    import time
    #numpy.random.seed(0)
    import mlp

    pat = [
        [[-1,-1], [0]],
        [[-1,1], [1]],
        [[1,-1], [1]],
        [[1,1], [0]]
    ]

    # Create a network with two input, two hidden, and one output nodes
    layers = [
                SOM(2, 4),
                basis.GaussianTransfer(),
                #mlp.Perceptron(4, 1, True),
                basis.GaussianOutput(4, 2, learn_rate=1.0),
                #mlp.Perceptron(4, 2, True),
                #mlp.SigmoidTransfer(),
                #mlp.Perceptron(2, 1),
                #mlp.SigmoidTransfer(),
             ]
    n = network.Network(layers)
    #print n.activate(pat[2][0])
    # Train it with some patterns
    start = time.clock()
    #n.logging = False
    n.train(pat, 1000, 0.02)
    print time.clock() - start
    # test it
    n.test(pat)
    print n._layers[0]._weights

if __name__ == '__main__':
    train_test()