"""Radial Basis Function network."""
import numpy

from learning import Model
from learning import SOM
from learning.architecture import transfer

INITIAL_WEIGHTS_RANGE = 0.25

class RBF(Model):
    """Radial Basis Function network."""
    def __init__(self, attributes, num_clusters, num_outputs,
                 learn_rate=1.0, variance=None, scale_by_similarity=True,
                 pre_train_clusters=False,
                 move_rate=0.1, neighborhood=2, neighbor_move_rate=1.0):
        super(RBF, self).__init__()

        # Clustering algorithm
        self._pre_train_clusters = pre_train_clusters
        self._som = SOM(
            attributes, num_clusters,
            move_rate=move_rate, neighborhood=neighborhood, neighbor_move_rate=neighbor_move_rate)

        # Variance for gaussian
        if variance is None:
            variance = 4.0/num_clusters
        self._variance = variance

        # Weight matrix for output
        self._weight_matrix = self._random_weight_matrix((num_clusters, num_outputs))
        self._learning_rate = learn_rate

        # Optional scaling output by total gaussian similarity
        self._scale_by_similarity = scale_by_similarity

        # For training
        self._similarities = None
        self._total_similarity = None

    def reset(self):
        """Reset this model."""
        self._som.reset()
        self._weight_matrix = self._random_weight_matrix(self._weight_matrix.shape)

        self._similarities = None
        self._total_similarity = None

    def _random_weight_matrix(self, shape):
        """Return a random weight matrix."""
        # TODO: Random weight matrix should be a function user can pass in
        return (2*numpy.random.random(shape) - 1)*INITIAL_WEIGHTS_RANGE

    def activate(self, inputs):
        """Return the model outputs for given inputs."""
        # Get distance to each cluster center, and apply guassian for similarity
        self._similarities = transfer.gaussian(self._som.activate(inputs), self._variance)

        # Get output by weighted summation of similarities, weighted by weights
        output = numpy.dot(self._similarities, self._weight_matrix)

        if self._scale_by_similarity:
            self._total_similarity = numpy.sum(self._similarities)
            output /= self._total_similarity

        return output

    def train(self, *args, **kwargs):
        """Train model to converge on a dataset.

        Note: Override this method for batch learning models.

        Args:
            input_matrix: A matrix with samples in rows and attributes in columns.
            target_matrix: A matrix with samples in rows and target values in columns.
            iterations: Max iterations to train model.
            retries: Number of times to reset model and retries if it does not converge.
                Convergence is defined as reaching error_break.
            error_break: Training will end once error is less than this.
            pattern_select_func: Function that takes (input_matrix, target_matrix),
                and returns a selection of rows. Use partial function to embed arguments.
        """
        if self._pre_train_clusters:
            # Train SOM first
            self._som.train(*args, **kwargs)

        super(RBF, self).train(*args, **kwargs)


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
        error_vec = output - target_vec
        error = numpy.mean(error_vec**2)

        if self._scale_by_similarity:
            error_vec /= self._total_similarity

        # Update weights weight gradient descent
        # TODO: Take an optimizer, instead of hard coding gradient descent
        # TODO: Line search for step size, instead of just self._learning_rate
        jacobian = self._similarities[:, None].dot(error_vec[None, :])
        self._weight_matrix -= self._learning_rate*jacobian

        return error
