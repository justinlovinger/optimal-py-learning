"""Radial Basis Function network."""
import numpy

from learning import Model
from learning import SOM
from learning import calculate
from learning.optimize import Problem, SteepestDescent

INITIAL_WEIGHTS_RANGE = 0.25

class RBF(Model):
    """Radial Basis Function network."""
    def __init__(self, attributes, num_clusters, num_outputs,
                 optimizer=None, variance=None, scale_by_similarity=True,
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

        # Optimizer to optimize weight_matrix
        if optimizer is None:
            optimizer = SteepestDescent()
        self._optimizer = optimizer

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
        self._similarities = calculate.gaussian(self._som.activate(inputs), self._variance)

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
        error, flat_weights = self._optimizer.next(
            Problem(obj_func=lambda xk: self._get_obj(xk, input_matrix, target_matrix),
                    obj_jac_func=lambda xk: self._get_obj_jac(xk, input_matrix, target_matrix)),
            self._weight_matrix.ravel()
        )
        self._weight_matrix = flat_weights.reshape(self._weight_matrix.shape)

        # Train SOM clusters
        self._som.train_step(input_matrix, target_matrix)

        return error

    def _get_obj(self, flat_weights, input_matrix, target_matrix):
        """Helper function for Optimizer."""
        self._weight_matrix = flat_weights.reshape(self._weight_matrix.shape)
        # TODO: Should take user provided error function
        return 0.5*self.avg_mse(input_matrix, target_matrix) 

    def _get_obj_jac(self, flat_weights, input_matrix, target_matrix):
        """Helper function for Optimizer."""
        self._weight_matrix = flat_weights.reshape(self._weight_matrix.shape)
        error, jacobian = self._get_jacobian(input_matrix, target_matrix)
        return error, jacobian.ravel()

    def _get_jacobian(self, input_matrix, target_matrix):
        """Return jacobian and error for given dataset."""
        errors, jacobians = zip(*[self._get_sample_jacobian(input_vec, target_vec)
                                  for input_vec, target_vec in zip(input_matrix, target_matrix)])
        return 0.5*numpy.mean(errors), numpy.mean(jacobians, axis=0)

    def _get_sample_jacobian(self, input_vec, target_vec):
        """Return jacobian and error for given sample."""
        output = self.activate(input_vec)
        # TODO: Should be based on user provided error function
        error_vec = output - target_vec
        error = numpy.mean(error_vec**2)

        if self._scale_by_similarity:
            error_vec /= self._total_similarity

        jacobian = self._similarities[:, None].dot(error_vec[None, :])

        return error, jacobian

    def _gradient_descent(self, jacobian):
        # Update weights weight gradient descent
        # TODO: Take an optimizer, instead of hard coding gradient descent
        # TODO: Line search for step size, instead of just self._learning_rate
        self._weight_matrix -= self._learning_rate*jacobian
