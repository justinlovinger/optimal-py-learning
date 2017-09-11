"""Radial Basis Function network."""
import numpy

from learning import Model
from learning import SOM
from learning.architecture import mlp
from learning.architecture import transfer

class RBF(Model):
    """Radial Basis Function network."""
    def __init__(self, attributes, num_clusters, num_outputs,
                 learn_rate=1.0, variance=None, scale_by_similarity=True,
                 move_rate=0.1, neighborhood=2, neighbor_move_rate=1.0):
        super(RBF, self).__init__()

        # Clustering algorithm
        self._som = SOM(
            attributes, num_clusters,
            move_rate=move_rate, neighborhood=neighborhood, neighbor_move_rate=neighbor_move_rate)

        # Variance for gaussian
        if variance is None:
            variance = 4.0/num_clusters
        self._variance = variance

        # Single layer perceptron for output
        self._perceptron = mlp.Perceptron(num_clusters, num_outputs, learn_rate=learn_rate)

        # Optional scaling output by total guassian similarity
        self._scale_by_similarity = scale_by_similarity

        # For training
        self._similarities = None
        self._total_similarity = None

    def reset(self):
        """Reset this model."""
        self._som.reset()
        self._perceptron.reset()

        self._similarities = None
        self._total_similarity = None

    def activate(self, inputs):
        """Return the model outputs for given inputs."""
        # Get distance to each cluster center, and apply guassian for similarity
        self._similarities = transfer.gaussian(self._som.activate(inputs), self._variance)

        # Get output from perceptron
        output = self._perceptron.activate(self._similarities)
        #self._output = output[:]
        if self._scale_by_similarity:
            self._total_similarity = numpy.sum(self._similarities)
            output /= self._total_similarity

        return output

    def train_step(self, input_matrix, target_matrix):
        """Adjust the model towards the targets for given inputs.

        Train on a mini-batch.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        # Train RBF
        error_vec = super(RBF, self).train_step(input_matrix, target_matrix)

        # Train SOM clusters
        self._som.train_step(input_matrix, target_matrix)

        return error_vec

    def _train_increment(self, input_vec, target_vec):
        """Train on a single input, target pair.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        output = self.activate(input_vec)
        error_vec = target_vec - output

        if self._scale_by_similarity:
            # NOTE: The math seems to say that we should divide by total_similarity
            # However, conceptually (and empirically) it makes more sense to multitpy
            #   If output is divided by a very small number (self._total_similarity)
            #   then it will become much larger, and any change in weight will be magnified
            #   therefore, we want make a small change in weights
            #   same thing vice versa
            error_vec *= self._total_similarity

        # Update perceptron
        # NOTE: Gradient is just error vector in this case
        self._perceptron.update(self._similarities, output, error_vec)

        return error_vec
