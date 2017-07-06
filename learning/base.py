import random
import pickle
import numbers

import numpy

##############################
# Pattern selection functions
##############################
def select_iterative(input_matrix, target_matrix):
    """Return all rows in order."""
    return input_matrix, target_matrix

def select_sample(input_matrix, target_matrix, size=None):
    """Return a random selection of rows, without replacement.

    Rows are returned in random order.
    Returns all rows in random order by default.
    """
    num_rows = input_matrix.shape[0]
    if size is None:
        size = num_rows

    selected_rows = random.sample(range(num_rows), size)
    return input_matrix[selected_rows], target_matrix[selected_rows]

def select_random(input_matrix, target_matrix, size=None):
    """Return a random selection of rows, with replacement.

    Rows are returned in random order.
    """
    num_rows = input_matrix.shape[0]
    if size is None:
        size = num_rows

    max_index = num_rows-1
    selected_rows = [random.randint(0, max_index) for _ in range(size)]
    return input_matrix[selected_rows], target_matrix[selected_rows]

class Model(object):
    """A supervised learning model."""
    def __init__(self):
        self._post_pattern_callback = None

        # Bookkeeping
        self.logging = True
        self.iteration = 0

    def _reset_bookkeeping(self):
        self.iteration = 0

    def reset(self):
        """Reset this model."""
        raise NotImplementedError()

    def activate(self, inputs):
        """Return the model outputs for given inputs."""
        raise NotImplementedError()

    def train(self, input_matrix, target_matrix,
              iterations=1000, retries=0, error_break=0.002,
              error_stagnant_distance=5, error_stagnant_threshold=0.00001,
              error_improve_iters=20,
              pattern_select_func=select_iterative, post_pattern_callback=None):
        """Train model to converge on a dataset.

        Note: Override this method for batch learning models.

        Args:
            input_matrix: A matrix with samples in rows and attributes in columns.
            target_matrix: A matrix with samples in rows and target values in columns.
            iterations: Max iterations to train model.
            retries: Number of times to reset model and retries if it does not converge.
                Convergence is defined as reaching error_break.
            error_break: Training will end once error is less than this.
            error_stagnant_distance: Number of iterations during which error must change by at least
                error_stagnant_threshold, or training ends.
            error_stagnant_threshold: Threshold by which error must change within
                error_stagnant_distance iterations, or training ends.
            error_improve_iters: Best error must decrease within this many iterations,
                or training ends.
            pattern_select_func: Function that takes (input_matrix, target_matrix),
                and returns a selection of rows. Use partial function to embed arguments.
        """
        # Make sure matrix parameters are np arrays
        self._reset_bookkeeping()
        self._post_pattern_callback = post_pattern_callback # For calling in other method

        # Initialize variables for retries
        best_try = (float('inf'), None) # (error, serialized_model)

        # Learn on each pattern for each iteration
        for attempt in range(retries+1):
            success = self._train_attempt(
                input_matrix, target_matrix,
                iterations, error_break, error_stagnant_distance, error_stagnant_threshold,
                error_improve_iters, pattern_select_func, post_pattern_callback)

            # Skip all the tracking and whatnot if there are no retries (optimization)
            if retries == 0:
                return

            # End if model converged
            # No need to use best attempt (since this is the first to reach best error)
            if success:
                return

            # TODO: Should we use user provided error function?
            attempt_error = self.avg_mse(input_matrix, target_matrix)

            # End when out of retries, use best attempt so far
            if attempt >= retries:
                if attempt_error < best_try[0]:
                    # Last attempt was our best
                    return
                else:
                    # Use best attempt
                    best_model = self.unserialize(best_try[1])
                    self.__dict__ = best_model.__dict__
                    return

            # Keep track of best attempt
            if attempt_error < best_try[0]:
                best_try = (attempt_error, self.serialize())

            # Reset for next attempt
            self.reset()

    def _train_attempt(self, input_matrix, target_matrix,
                       iterations, error_break, error_stagnant_distance, error_stagnant_threshold,
                       error_improve_iters, pattern_select_func, post_pattern_callback):
        """Attempt to train this model.

        Return True if model converged (error <= error_break)
        """
        # Initialize error history with errors that are
        # unlikely to be close in reality
        error_history = [1e10]*error_stagnant_distance

        # Initialize best error for error_decrease_iters
        best_error = float('inf')
        iters_since_improvement = 0

        for self.iteration in range(1, iterations+1):
            selected_patterns = pattern_select_func(input_matrix, target_matrix)

            # Learn each selected pattern
            error = self.train_step(*selected_patterns)

            # Logging and breaking
            if self.logging:
                print "Iteration {}, Error: {}".format(self.iteration, error)

            if error is not None:
                # Break early to prevent overtraining
                if (error <= error_break
                        # Perform a second test on whole dataset
                        # incase model is training on mini-batches
                        # TODO: Should we use user provided error function?
                        and self.avg_mse(input_matrix, target_matrix) <= error_break):
                    return True

                # Skip the rest if we're already out of iterations (optimization)
                # Useful for situations where we only run 1 iteration
                if self.iteration == iterations:
                    return False

                # Break if no progress is made
                if _all_close(error_history, error, error_stagnant_threshold):
                    # Break if not enough difference between all resent errors
                    # and current error
                    return False
                error_history.append(error)
                error_history.pop(0)

                # Break if best error has not improved within n iterations
                # Keep track of best error, and iterations since best error has improved
                if error < best_error:
                    best_error = error
                    iters_since_improvement = 0
                else:
                    iters_since_improvement += 1
                # If it has been too many iterations since improvement, break
                if iters_since_improvement >= error_improve_iters:
                    return False

        return False


    def train_step(self, input_matrix, target_matrix):
        """Adjust the model towards the targets for given inputs.

        Train on a mini-batch.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        # Learn each selected pattern
        error = 0.0
        for input_vec, target_vec in zip(input_matrix, target_matrix):
            # Learn
            next_error = self._train_increment(input_vec, target_vec)

            # Validate error
            if not isinstance(next_error, (numbers.Number, type(None))):
                raise TypeError('%s._train_increment must return an error number or None' % type(self))

            # Optional callback for user extension,
            # such as a visualization or history tracking
            if self._post_pattern_callback:
                self._post_pattern_callback(self, input_vec, target_vec)

            # Sum errors
            try:
                error += next_error
            except TypeError:
                # train_step doesn't return error
                error = None

        # Logging and breaking
        try:
            return error / len(input_matrix)
        except TypeError:
            # _train_increment doesn't return error
            return None

    def _train_increment(self, input_vec, target_vec):
        """Train on a single input, target pair.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        raise NotImplementedError()

    def serialize(self):
        """Convert model into string.

        Optional: Override for custom serialization.
        Defaults to pickle, protocol 2.

        Returns:
            string; A string representing this network.
        """
        return pickle.dumps(self, protocol=2)

    @classmethod
    def unserialize(cls, serialized_model):
        """Convert serialized model into Model.

        Optional: Override for custom serialization.

        Returns:
            Model; A Model object.
        """
        model = pickle.loads(serialized_model)
        if type(model) != cls:
            raise ValueError('serialized_model does not match this class')
        return model

    ##################
    # Helper methods
    ##################
    def test(self, input_matrix, target_matrix):
        """Print corresponding inputs and outputs from a dataset."""
        for inp_vec, tar_vec in zip(input_matrix, target_matrix):
            print(tar_vec, '->', self.activate(inp_vec))

    def avg_mse(self, input_matrix, target_matrix):
        """Return the average mean squared error for a dataset."""
        error = 0.0
        for input_vec, target_vec in zip(input_matrix, target_matrix):
            error = error + self.mse(input_vec, target_vec)

        return error/len(input_matrix)

    def mse(self, input_vec, target_vec):
        """Return the mean squared error (MSE) for a pattern."""
        # Mean squared error
        return numpy.mean(numpy.subtract(self.activate(input_vec), target_vec)**2)


def _all_close(values, other_value, threshold):
    """Return true if all values are within threshold distance of other_value."""
    for value in values:
        if abs(value - other_value) > threshold:
            return False
    return True
