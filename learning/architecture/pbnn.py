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

import numpy

from learning import calculate
from learning import Model
from learning import calculate


class PBNN(Model):
    def __init__(self,
                 variance=None,
                 scale_by_similarity=True,
                 scale_by_class=True):
        super(PBNN, self).__init__()

        if variance is None:
            # TODO: Adjust it during training
            self._variance = 1.0
        else:
            self._variance = variance
        self._scale_by_class = scale_by_class
        self._scale_by_similarity = scale_by_similarity

        self._input_matrix = None  # Inputs stored when training
        self._target_matrix = None  # Targets stored when training
        self._target_totals = None  # Sum of rows in target matrix

    def reset(self):
        """Reset this model."""
        super(PBNN, self).reset()

        self._input_matrix = None
        self._target_matrix = None
        self._target_totals = None

    def activate(self, inputs):
        """Return the model outputs for given inputs."""
        # Calculate similarity between input and each stored input
        # (gaussian of each distance)
        similarities = calculate.gaussian(
            _distances(inputs, self._input_matrix), self._variance)
        # Then scale each stored target by corresponding similarity, and sum
        output_vec = _weighted_sum_rows(self._target_matrix, similarities)

        if self._scale_by_similarity:
            output_vec /= numpy.sum(similarities)

        if self._scale_by_class:
            # Scale output by number of classes (sum of targets)
            # This minimizes the effect of unbalanced classes
            # Return 0 when target total is 0
            output_vec[:] = calculate.protvecdiv(output_vec,
                                                 self._target_totals)

        # Convert output to probabilities, and return
        output_vec /= sum(output_vec)
        return output_vec

    def train(self, input_matrix, target_matrix, *args, **kwargs):
        # Store inputs to recall later
        self._input_matrix = numpy.copy(input_matrix)

        # Store targets to recall later
        self._target_matrix = numpy.copy(target_matrix)

        # Calculate target sum now, for efficiency
        self._target_totals = numpy.sum(self._target_matrix, axis=0)


def _distances(x_vec, y_matrix):
    """Return vector of distances between x_vec and each y_matrix row."""
    diffs = x_vec - y_matrix
    distances = [numpy.sqrt(d.dot(d)) for d in diffs]
    return numpy.array(distances)


def _weighted_sum_rows(x_matrix, scaling_vector):
    """Return sum of rows in x_matrix, each row scaled by scalar in scaling_vector."""
    return numpy.sum(x_matrix * scaling_vector[:, numpy.newaxis], axis=0)
