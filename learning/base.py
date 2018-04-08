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
"""Base model and functions for learning methods."""

import math
import random
import pickle
import numbers

import numpy

from learning import validation


##############################
# Pattern selection functions
##############################
def select_sample(input_matrix, target_matrix, size=None):
    """Return a random selection of rows, without replacement.

    Rows are returned in random order.
    Returns all rows in random order by default.
    """
    num_rows = input_matrix.shape[0]
    if size is None:
        size = _selection_size_heuristic(input_matrix.shape[0])

    selected_rows = random.sample(range(num_rows), size)
    return input_matrix[selected_rows], target_matrix[selected_rows]


def select_random(input_matrix, target_matrix, size=None):
    """Return a random selection of rows, with replacement.

    Rows are returned in random order.
    """
    num_rows = input_matrix.shape[0]
    if size is None:
        size = _selection_size_heuristic(input_matrix.shape[0])

    max_index = num_rows - 1
    selected_rows = [random.randint(0, max_index) for _ in range(size)]
    return input_matrix[selected_rows], target_matrix[selected_rows]


def _selection_size_heuristic(num_samples):
    """Return size of mini-batch, given size of dataset."""
    # Incrase size of mini-batch gradually as number of samples increases,
    # with a soft cap (using log).
    # But do not return size larger than number of samples
    return min(num_samples,
               int(20.0 *
                   (math.log(num_samples + 10, 2) - math.log(10.0) + 1.0)))


class Model(object):
    """A supervised learning model."""

    def __init__(self):
        self._post_pattern_callback = None

        # Bookkeeping
        self.logging = True
        self.iteration = 0
        self.converged = False

    def _reset_bookkeeping(self):
        self.iteration = 0

    def reset(self):
        """Reset this model."""
        self.iteration = 0
        self.converged = False

    def activate(self, input_vec):
        """Return the model outputs for given inputs."""
        raise NotImplementedError()

    def stochastic_train(self,
                         input_matrix,
                         target_matrix,
                         max_iterations=100,
                         error_break=0.002,
                         pattern_selection_func=select_sample,
                         train_kwargs={'iterations': 100}):
        """Train model on multiple subsets of the given dataset.

        Use for stochastic gradient descent.

        Args:
            input_matrix: A matrix with samples in rows and attributes in columns.
            target_matrix: A matrix with samples in rows and target values in columns.
            max_iterations: Maximum number of times that Model.train is called.
            error_break: Training will end once error is less than this, on entire dataset.
            pattern_select_func: Function that takes (input_matrix, target_matrix),
                and returns a selection of rows. Use partial function to embed arguments.
        """
        for iteration in range(1, max_iterations + 1):
            train_error = self.train(*pattern_selection_func(
                input_matrix, target_matrix), **train_kwargs)

            if self.converged:
                # Break early to prevent overtraining
                if (train_error <= error_break
                        # Perform a second test on whole dataset
                        # TODO: Use user provided error function
                        and validation.get_error(
                            self, input_matrix, target_matrix) <= error_break):
                    return train_error

        # Override iteration from inner loop, with iteration number from outer loop
        self.iteration = iteration

        return train_error

    def train(self,
              input_matrix,
              target_matrix,
              iterations=1000,
              retries=0,
              error_break=0.002,
              error_stagnant_distance=5,
              error_stagnant_threshold=0.00001,
              error_improve_iters=20,
              post_pattern_callback=None):
        """Train model on the given dataset.

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
        """
        # Even if we don't reset, users will expect the Model to train if train is called
        # So we reset self.converged
        self.converged = False

        # Pre training callback
        self._pre_train(input_matrix, target_matrix)

        # Train, with given arguments
        train_error = self._train(
            input_matrix, target_matrix, iterations, retries, error_break,
            error_stagnant_distance, error_stagnant_threshold,
            error_improve_iters, post_pattern_callback)

        # Post training callback
        self._post_train(input_matrix, target_matrix)

        return train_error


    def _train(self, input_matrix, target_matrix, iterations, retries,
               error_break, error_stagnant_distance, error_stagnant_threshold,
               error_improve_iters, post_pattern_callback):
        """Train model on the given dataset."""
        self._reset_bookkeeping()
        self._post_pattern_callback = post_pattern_callback  # For calling in other method

        # Initialize variables for retries
        best_try = (float('inf'), None)  # (error, serialized_model)

        # Learn on each pattern for each iteration
        for attempt in range(retries + 1):
            attempt_error = self._train_attempt(
                input_matrix, target_matrix, iterations, error_break,
                error_stagnant_distance, error_stagnant_threshold,
                error_improve_iters, post_pattern_callback)

            # End if model converged
            # No need to use best attempt (since this is the first to reach best error)
            if self.converged:
                return attempt_error

            # Skip all the tracking and whatnot if there are no retries (optimization)
            if retries == 0:
                return attempt_error

            # End when out of retries, use best attempt so far
            if attempt >= retries:
                if attempt_error < best_try[0]:
                    # Last attempt was our best
                    return attempt_error
                else:
                    # Use best attempt
                    best_model = self.unserialize(best_try[1])
                    self.__dict__ = best_model.__dict__
                    return attempt_error

            # Keep track of best attempt
            if attempt_error < best_try[0]:
                best_try = (attempt_error, self.serialize())

            # Reset for next attempt
            self.reset()

        return attempt_error

    def _train_attempt(self, input_matrix, target_matrix, iterations,
                       error_break, error_stagnant_distance,
                       error_stagnant_threshold, error_improve_iters,
                       post_pattern_callback):
        """Attempt to train this model.

        Return True if model converged (error <= error_break)
        """
        # Initialize error history with errors that are
        # unlikely to be close in reality
        error_history = [1e10] * error_stagnant_distance

        # Initialize best error for error_decrease_iters
        best_error = float('inf')
        iters_since_improvement = 0

        for self.iteration in range(1, iterations + 1):
            # Perform a single training step
            error = self.train_step(input_matrix, target_matrix)

            # Logging and breaking
            if self.logging:
                print "Iteration {}, Error: {}".format(self.iteration, error)

            if self.converged:  # If model set converged with custom criteria
                return error

            if error is not None:
                # Break if error is sufficient, useful to prevent overfitting
                if error <= error_break:
                    self.converged = True
                    return error

                # Skip the rest if we're already out of iterations (optimization)
                # Useful for situations where we only run 1 iteration
                if self.iteration == iterations:
                    return error

                # Break if no progress is made
                if _all_close(error_history, error, error_stagnant_threshold):
                    # Break if not enough difference between all resent errors
                    # and current error
                    return error
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
                    return error

        return error

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
                raise TypeError(
                    '%s._train_increment must return an error number or None' %
                    type(self))

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

    def _pre_train(self, input_matrix, target_matrix):
        """Call before Model.train.

        Optional.
        """
        pass

    def _post_train(self, input_matrix, target_matrix):
        """Call after Model.train.
        
        Optional.
        """
        pass

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
    def print_results(self, input_matrix, target_matrix):
        """Print corresponding inputs and outputs from a dataset."""
        for inp_vec, tar_vec in zip(input_matrix, target_matrix):
            print inp_vec, '->', self.activate(inp_vec), '(%s)' % tar_vec


def _all_close(values, other_value, threshold):
    """Return true if all values are within threshold distance of other_value."""
    for value in values:
        if abs(value - other_value) > threshold:
            return False
    return True
