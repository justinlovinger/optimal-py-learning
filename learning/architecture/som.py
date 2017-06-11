import operator

import numpy

from learning import Model
from learning import calculate

class SOM(Model):
    def __init__(self, attributes, neurons, 
                 move_rate=0.1, neighborhood=2, neighbor_move_rate=1.0,
                 initial_weights_range=1.0):
        super(SOM, self).__init__()

        self.move_rate = move_rate
        self.neighborhood = neighborhood
        self.neighbor_move_rate = neighbor_move_rate
        self.initial_weights_range = initial_weights_range

        self._size = (neurons, attributes)
        self._weights = numpy.zeros(self._size)
        self._distances = numpy.zeros(neurons)

        self.reset()

    def reset(self):
        """Reset this model."""
        # Randomize weights, between -1 and 1
        self._weights = (2*numpy.random.random(self._size) - 1)*self.initial_weights_range
        self._distances = numpy.zeros(self._size)

    def activate(self, inputs):
        """Return the model outputs for given inputs."""
        diffs = inputs - self._weights
        self._distances = [numpy.sqrt(d.dot(d)) for d in diffs]
        return numpy.array(self._distances)

    def _train_increment(self, input_vec, target_vec):
        """Train on a single input, target pair.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        self.activate(input_vec)
        self._move_neurons(input_vec)

    def _move_neurons(self, input_vec):
        # Perform a competition, and move the winner closer to the input
        closest = self._get_closest()

        # Move the winner and neighbors closer
        # The further the neighbor, the less it should move
        for i in range(closest-self.neighborhood, closest+self.neighborhood+1):
            if i >= 0 and i < self._size[0]: # if in range
                neighbor_distance = float(abs(i-closest))
                move_rate_modifier = calculate.gaussian(neighbor_distance,
                                                       self.neighbor_move_rate)
                final_rate = move_rate_modifier*self.move_rate

                self._weights[i] += final_rate*(input_vec-self._weights[i])

    def _get_closest(self):
        return _min_index(self._distances)

def _min_index(values):
    return min(enumerate(values), key=operator.itemgetter(1))[0]
