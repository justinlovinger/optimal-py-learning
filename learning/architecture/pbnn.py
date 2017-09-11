import numpy

from learning import Model
from learning.architecture import transfer

class PBNN(Model):
    def __init__(self, variance=None, scale_by_similarity=True, scale_by_class=True):
        super(PBNN, self).__init__()

        if variance is None:
            # TODO: Adjust it during training
            self._variance = 1.0
        else:
            self._variance = variance
        self._scale_by_class = scale_by_class
        self._scale_by_similarity = scale_by_similarity

        self._input_matrix = None # Inputs stored when training
        self._target_matrix = None # Targets stored when training
        self._target_totals = None # Sum of rows in target matrix

    def reset(self):
        """Reset this model."""
        self._input_matrix = None
        self._target_matrix = None
        self._target_totals = None

    def activate(self, inputs):
        """Return the model outputs for given inputs."""
        # Calculate similarity between input and each stored input
        # (gaussian of each distance)
        similarities = transfer.gaussian(_distances(inputs, self._input_matrix), self._variance)
        # Then scale each stored target by corresponding similarity, and sum
        output_vec = _weighted_sum_rows(self._target_matrix, similarities)

        if self._scale_by_similarity:
            output_vec /= numpy.sum(similarities)

        if self._scale_by_class:
            # Scale output by number of classes (sum of targets)
            # This minimizes the effect of unbalanced classes
            output_vec /= self._target_totals

        # Convert output to probabilities, and return
        output_vec /= sum(output_vec)
        return output_vec

    def train(self, patterns):
        # Extract inputs from patterns
        # And store them to recall later
        inputs = [p[0] for p in patterns]
        self._input_matrix = numpy.array(inputs)

        # Extract targets from patterns
        # And store them to recall later
        targets = [p[1] for p in patterns]
        self._target_matrix = numpy.array(targets)

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
