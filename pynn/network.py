import numpy
import random

##############################
# Pattern selection functions
##############################
def select_iterative(patterns):
    return patterns

def select_sample(patterns, size=None):
    if size == None:
        size = len(patterns)

    return random.sample(patterns, size)

def select_random(patterns, size=None):
    if size == None:
        size = len(patterns)

    max_index = len(patterns)-1
    return [patterns[random.randint(0, max_index)] for i in range(size)]

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

    def train(self, patterns, iterations=1000, error_break=0.002,
              error_stagnant_distance=5, error_stagnant_threshold=0.0001,
              pattern_select_func=select_iterative, post_pattern_callback=None):
        """Train model to converge on set of patterns.

        Note: Override this method for batch learning models.

        Args:
            patterns: A set of (inputs, targets) pairs.
            iterations: Max iterations to train network.
            error_break: Training will end once error is less than this.
            pattern_select_func: Function that takes a set of patterns,
                and returns a set of patterns. Use partial function to embed arguments.
        """
        self._reset_bookkeeping()
        self._post_pattern_callback = post_pattern_callback # For calling in other method

        # Initialize error history with errors that are
        # unlikey to be close in reality
        error_history = [1e10]*error_stagnant_distance

        # Learn on each pattern for each iteration
        for self.iteration in range(1, iterations+1):
            self.pre_iteration(patterns)

            # Learn each selected pattern
            selected_patterns = pattern_select_func(patterns)
            input_matrix = numpy.array([p[0] for p in selected_patterns])
            target_matrix = numpy.array([p[1] for p in selected_patterns])
            error = self.train_step(input_matrix, target_matrix)

            self.post_iteration(patterns)

            # Logging and breaking
            if self.logging:
                print "Iteration {}, Error: {}".format(self.iteration, error)

            if error is not None:
                # Break early to prevent overtraining
                if error < error_break:
                    break

                # Break if no progress is made
                if _all_close(error_history, error, error_stagnant_threshold):
                    # Break if not enough difference between all resent errors
                    # and current error
                    break

                error_history.append(error)
                error_history.pop(0)

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
            errors = self._train_increment(input_vec, target_vec)

            # Optional callback for user extension,
            # such as a visualization or history tracking
            if self._post_pattern_callback:
                self._post_pattern_callback(self, input_vec, target_vec)

            # Sum errors
            try:
                error += numpy.mean(errors**2)
            except TypeError:
                # train_step doesn't return error
                error = None

        # Logging and breaking
        try:
            return error / input_matrix.shape[0]
        except TypeError:
            # _train_increment doesn't return error
            return None

    def _train_increment(self, input_vec, target_vec):
        """Train on a single input, target pair.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        raise NotImplementedError()

    def pre_iteration(self, patterns):
        """Optional. Callback performed before each training iteration.

        Note: If self.train is overwritten, this may not be called.
        """
        pass

    def post_iteration(self, patterns):
        """Optional. Callback performed after each training iteration.

        Note: If self.train is overwritten, this may not be called.
        """
        pass

    def serialize(self):
        """Convert model into string.

        Returns:
            string; A string representing this network.
        """
        raise NotImplementedError()

    @classmethod
    def unserialize(cls, serialized_model):
        """Convert serialized model into Model.

        Returns:
            Model; A Model object.
        """
        raise NotImplementedError()

    ##################
    # Helper methods
    ##################
    def test(self, patterns):
        """Print corresponding inputs and outputs from a set of patterns."""
        for p in patterns:
            print(p[0], '->', self.activate(p[0]))

    def mse(self, pattern):
        """Return the mean squared error (MSE) for a pattern."""
        # Mean squared error
        return numpy.mean(numpy.subtract(self.activate(pattern[0]), pattern[1])**2)

    def avg_mse(self, patterns):
        """Return the average mean squared error for a set of patterns."""
        error = 0.0
        for pattern in patterns:
            error = error + self.mse(pattern)

        return error/len(patterns)

def _all_close(values, other_value, threshold):
    """Return true if all values are within threshold distance of other_value."""
    for value in values:
        if abs(value - other_value) > threshold:
            return False
    return True
