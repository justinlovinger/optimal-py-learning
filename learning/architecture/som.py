###############################################################################
# The MIT License (MIT)
#
# Copyright (c) 2017 Justin Lovinger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import operator

import numpy

from learning import Model
from learning import calculate


class SOM(Model):
    def __init__(self,
                 attributes,
                 neurons,
                 move_rate=0.1,
                 neighborhood=2,
                 neighbor_move_rate=1.0,
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
        self._weights = (2 * numpy.random.random(self._size) - 1
                         ) * self.initial_weights_range
        self._distances = numpy.zeros(self._size)

    def activate(self, input_tensor):
        """Return the model outputs for given input_tensor."""
        if not isinstance(input_tensor, numpy.ndarray):
            input_tensor = numpy.array(input_tensor)

        if len(input_tensor.shape) == 1:
            diff_matrix = input_tensor - self._weights
            # Dot each row of diffs with itself (a.k.a. numpy.sum(diffs**2, axis=-1))
            # Then sqrt result
            self._distances = numpy.sqrt(
                numpy.einsum('ij,ij->i', diff_matrix, diff_matrix))
        elif len(input_tensor.shape) == 2:
            # Reshape input_tensor to obtain the difference between
            # each row of input_tensor and self._weights
            diff_tensor = input_tensor.reshape(input_tensor.shape[0], 1,
                                         input_tensor.shape[1]) - self._weights
            # Dot each row of diffs with itself (a.k.a. numpy.sum(diffs**2, axis=-1))
            # Then sqrt result
            # For each difference matrix in diff_tensor
            self._distances = numpy.sqrt(
                numpy.einsum('ijk,ijk->ij', diff_tensor, diff_tensor))
        else:
            raise ValueError('Invalid shape of input_tensor.')

        return self._distances

    def _train_increment(self, input_vec, target_vec):
        """Train on a single input, target pair.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        # TODO (maybe): Train on bath of input_matrix target_matrix,
        # instead of one vector at a time.
        # This will change the behavior of training, but will be more efficient.
        self.activate(input_vec)
        self._move_neurons(input_vec)

    def _move_neurons(self, input_vec):
        # Perform a competition, and move the winner closer to the input
        closest = numpy.argmin(self._distances)

        # Move the winner and neighbors closer
        # The further the neighbor, the less it should move
        for i in range(closest - self.neighborhood,
                       closest + self.neighborhood + 1):
            if i >= 0 and i < self._size[0]:  # if in range
                neighbor_distance = float(abs(i - closest))
                move_rate_modifier = calculate.gaussian(
                    neighbor_distance, self.neighbor_move_rate)
                final_rate = move_rate_modifier * self.move_rate

                self._weights[i] += final_rate * (input_vec - self._weights[i])
