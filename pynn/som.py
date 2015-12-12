import numpy

from pynn import network
from pynn import transfer

def distance(vec_a, vec_b):
    diff = numpy.subtract(vec_a, vec_b)
    return diff.dot(diff)

class SOM(network.Layer):
    requires_prev = (None,)

    def __init__(self, inputs, neurons, 
                 move_rate=0.1, neighborhood=2, neighbor_move_rate=1.0):
        super(SOM, self).__init__()

        self.move_rate = move_rate
        self.neighborhood = neighborhood
        self.neighbor_move_rate = neighbor_move_rate

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

    def get_prev_errors(self, errors, outputs):
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

    def move_neurons(self, inputs):
        # Perform a competition, and move the winner closer to the input
        closest = self.get_closest()

        # Move the winner and neighbors closer
        # The further the neighbor, the less it should move
        for i in range(closest-self.neighborhood, closest+self.neighborhood+1):
            if i >= 0 and i < self._size[0]: # if in range
                neighbor_distance = float(abs(i-closest))
                move_rate_modifier = transfer.gaussian(neighbor_distance, 
                                                       self.neighbor_move_rate)
                final_rate = move_rate_modifier*self.move_rate

                self._weights[i] += final_rate*(inputs-self._weights[i])

    def update(self, inputs, outputs, deltas):
        self.move_neurons(inputs)