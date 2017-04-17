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

    def train_step(self, inputs, targets):
        """Adjust the model towards the targets for given inputs.

        Optional.
        Only for incremental learning models.
        """
        raise NotImplementedError()

    def train(self, patterns, iterations=1000, error_break=0.002,
              error_stagnant_distance=5, error_stagnant_threshold=0.0001,
              pattern_select_func=select_iterative, post_pattern_callback=None,
              preprocess_func=None):
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

        # Preprocess data
        if preprocess_func is not None:
            patterns = preprocess_func(patterns)

        # Initialize error history with errors that are
        # unlikey to be close in reality
        error_history = [1e10]*error_stagnant_distance

        # Learn on each pattern for each iteration
        for self.iteration in range(1, iterations+1):

            # Learn each selected pattern
            error = 0.0
            for pattern in pattern_select_func(patterns):
                # Learn
                errors = self.train_step(pattern[0], pattern[1])

                # Optional callback for user extension,
                # such as a visualization or history tracking
                if post_pattern_callback:
                    post_pattern_callback(self, pattern)

                # Sum errors
                try:
                    error += numpy.mean(errors**2)
                except TypeError:
                    # train_step doesn't return error
                    error = None

            # Logging and breaking
            try:
                error = error / len(patterns)
            except TypeError:
                # train_step doesn't return error
                error = None
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

##########################
# Quick network functions
##########################
def make_mlp_classifier(shape, learn_rate=0.5, momentum_rate=0.1):
    """Create a multi-layer perceptron network for classification."""
    from pynn.architecture import mlp

    layers = _make_mlp(shape, learn_rate, momentum_rate)

    # Softmax for classification
    layers.append(mlp.SoftmaxTransferPerceptron())

    return Network(layers)

def make_dropout_mlp_classifier(shape, learn_rate=0.5, momentum_rate=0.1,
                                input_active_probability=0.8, hidden_active_probability=0.5):
    """Create a multi-layer perceptron network with dropout for classification."""
    from pynn.architecture import mlp

    layers = _make_dropout_mlp(shape, learn_rate, momentum_rate,
                               input_active_probability, hidden_active_probability)

    # Softmax for classification
    layers.append(mlp.SoftmaxTransferPerceptron())

    return Network(layers)
