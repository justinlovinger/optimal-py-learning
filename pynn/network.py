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

def make_mlp(shape, learn_rate=0.5, momentum_rate=0.1):
    """Create a multi-layer perceptron network for regression or classification."""
    from pynn.architecture import mlp

    layers = _make_mlp(shape, learn_rate, momentum_rate)

    # Linear output for regression
    return Network(layers)

def _make_mlp(shape, learn_rate=0.5, momentum_rate=0.1):
    """Return the common layers in regression and classification mlps."""
    from pynn.architecture import mlp

    # Create first layer with bias
    layers = [mlp.AddBias(mlp.Perceptron(shape[0]+1, shape[1],
                                         learn_rate, momentum_rate)),
              mlp.ReluTransferPerceptron()]

    # After are hidden layers with given shape
    num_inputs = shape[1]
    for num_outputs in shape[2:-1]:
        # Add perceptron followed by transfer
        layers.append(mlp.Perceptron(num_inputs, num_outputs,
                                     learn_rate, momentum_rate))
        layers.append(mlp.ReluTransferPerceptron())

        num_inputs = num_outputs

    # Final transfer function must be able to output negatives and positives
    layers.append(mlp.Perceptron(shape[-2], shape[-1],
                                 learn_rate, momentum_rate))

    return layers


def make_dropout_mlp_classifier(shape, learn_rate=0.5, momentum_rate=0.1,
                                input_active_probability=0.8, hidden_active_probability=0.5):
    """Create a multi-layer perceptron network with dropout for classification."""
    from pynn.architecture import mlp

    layers = _make_dropout_mlp(shape, learn_rate, momentum_rate,
                               input_active_probability, hidden_active_probability)

    # Softmax for classification
    layers.append(mlp.SoftmaxTransferPerceptron())

    return Network(layers)


def make_dropout_mlp(shape, learn_rate=0.5, momentum_rate=0.1,
                     input_active_probability=0.8, hidden_active_probability=0.5):
    """Create a multi-layer perceptron network with dropout for regression or classification."""
    from pynn.architecture import mlp

    layers = _make_dropout_mlp(shape, learn_rate, momentum_rate,
                               input_active_probability, hidden_active_probability)

    # Linear output for regression
    return Network(layers)


def _make_dropout_mlp(shape, learn_rate, momentum_rate,
                      input_active_probability, hidden_active_probability):
    """Return the common layers in regression and classification dropout mlps."""
    from pynn.architecture import mlp

    # First layer is a special layer that disables inputs during training
    # Next is a special perceptron layer with bias (bias added by DropoutInputs)
    biased_perceptron = mlp.DropoutPerceptron(shape[0]+1, shape[1],
                                              learn_rate, momentum_rate,
                                              active_probability=hidden_active_probability)
    layers = [mlp.DropoutInputs(shape[0], input_active_probability),
              biased_perceptron, mlp.ReluTransferPerceptron()]

    # After are all other layers that make up the shape
    num_inputs = shape[1]
    for num_outputs in shape[2:-1]:
        # Add perceptron followed by transfer
        layers.append(mlp.DropoutPerceptron(num_inputs, num_outputs,
                                            learn_rate, momentum_rate,
                                            active_probability=hidden_active_probability))
        layers.append(mlp.ReluTransferPerceptron())

        num_inputs = num_outputs

    # Final transfer function must be able to output negatives and positives,
    # Last perceptron layer must not reduce number of outputs
    layers.append(mlp.DropoutPerceptron(shape[-2], shape[-1],
                                        learn_rate, momentum_rate,
                                        active_probability=1.0))

    return layers


def make_rbf(inputs, neurons, outputs, learn_rate=1.0, variance=None, normalize=True,
             move_rate=0.1, neighborhood=2, neighbor_move_rate=1.0,):
    """Create a radial-basis function network."""
    from pynn.architecture import transfer
    from pynn.architecture import mlp
    from pynn.architecture import som

    if variance == None:
        variance = 4.0/neurons

    som_ = som.SOM(inputs, neurons, move_rate, neighborhood, neighbor_move_rate)
    gaussian_transfer = transfer.GaussianTransfer(variance)
    perceptron = mlp.Perceptron(neurons, outputs, learn_rate, momentum_rate=0.0)
    
    layers = {'I': [som_],
              som_: [gaussian_transfer],
              gaussian_transfer: [perceptron]}

    if normalize:
        normalize_layer = transfer.NormalizeTransfer()
        layers[perceptron] = [normalize_layer]
        layers[gaussian_transfer].append(normalize_layer)
        layers[normalize_layer] = ['O']

        incoming_order_dict = {normalize_layer: [perceptron, gaussian_transfer]}
    else:
        layers[perceptron] = ['O']
        incoming_order_dict = None

    return Network(layers, incoming_order_dict)

def make_pbnn(variance=None):
    from pynn.architecture import pbnn
    from pynn.architecture import transfer

    # Calculate distances to stored points
    store_inputs = pbnn.StoreInputsLayer()
    distances = pbnn.DistancesLayer()

    # Gaussian transfer
    if variance is not None:
        gaussian_transfer = transfer.GaussianTransfer(variance=variance)
    else:
        gaussian_transfer = transfer.GaussianTransfer()

    # Weighted summation (weighted by output of guassian transfer), sums targets
    store_targets = pbnn.StoreTargetsLayer()
    weighted_summation = pbnn.WeightedSummationLayer()

    # TODO: Normalize by class counts
    # Normalize output to sum 1
    normalize_transfer = transfer.NormalizeTransfer()

    layers = {'I': [distances],
              store_inputs: [distances],
              distances: [gaussian_transfer],
              gaussian_transfer: [weighted_summation, normalize_transfer],
              store_targets: [weighted_summation],
              weighted_summation: [normalize_transfer],
              normalize_transfer: ['O']}

    incoming_order_dict = {normalize_transfer:[weighted_summation, gaussian_transfer],
                           distances: ['I', store_inputs],
                           weighted_summation: [gaussian_transfer, store_targets]}

    return Network(layers, incoming_order_dict)
