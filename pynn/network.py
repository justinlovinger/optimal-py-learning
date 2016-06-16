import numpy
import random

from pynn import graph

class Layer(object):
    """A layer of computation for a supervised learning network."""
    attributes = tuple([]) # Attributes for this layer

    requires_prev = tuple([]) # Attributes that are required in the previous layer
    requires_next = tuple([]) # Attributes that are required in the next layer

    def __init__(self, *args, **kwargs):
        self.network = None

    def reset(self):
        raise NotImplementedError()

    def activate(self, inputs):
        raise NotImplementedError()

    def _avg_all_errors(self, all_errors, expected_shape):
        # For efficiency, and because it is a common case
        if len(all_errors) == 1:
            return all_errors[0]

        # Avg all non None errors
        sum = numpy.zeros_like(all_errors[0])
        num_averaged = 0
        for errors in all_errors:
            if errors is not None and errors.shape == expected_shape:
                sum += errors
                num_averaged += 1

        if num_averaged == 0:
            # No errors in lsit
            return None
        else:
            return sum / num_averaged

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        raise NotImplementedError()

    def update(self, all_inputs, outputs, all_errors):
        raise NotImplementedError()

    def pre_training(self, patterns):
        """Called before each training run.
        
        Optional.
        """

    def post_training(self, patterns):
        """Called after each training run.
        
        Optional.
        """

    def pre_iteration(self, patterns):
        """Called before each training iteration.

        Optional.
        """

    def post_iteration(self, patterns):
        """Called after each training iteration.

        Optional.
        """

class SupportsGrowingLayer(Layer):
    """Layer that supports new neurons by increasing input size.
    
    Adds neurons when GrowingLayer before it raises growth exception.
    """
    attributes = ('supportsGrowing',)

    def add_input(self):
        raise NotImplementedError()

class GrowingLayer(Layer):
    """Layer that can add new neurons, increasing output size.
    
    Raise growth exception when new neuron is added.
    """
    requires_next = ('supportsGrowing',)

class ParallelLayer(Layer):
    """A composite of layers connected in parallel."""

#####################
# Network validation
#####################
def _validate_layers_layers(layers):
    """Assert each element of layers is a Layer."""
    for layer in layers:
        if not isinstance(layer, Layer):
            raise TypeError("layers argument of Network must contain Layer's." \
                             " Instead contains {}.".format(type(layer)))

def _all_in(all_required, values):
    for required in all_required:
        if not required in values:
            return False
    return True

def _validate_requires_next(layer, next_layer):
    if not _all_in(layer.requires_next, next_layer.attributes):
        raise TypeError("Layer of type {} must be followed by attributes: {}. " \
                        "It is followed by attributes: " \
                        "{}".format(type(layer), layer.requires_next, 
                                    next_layer.attributes))

def _validate_requires_prev(layer, prev_layer):
    if not _all_in(layer.requires_prev, prev_layer.attributes):
        raise TypeError("Layer of type {} must be preceded by attributes: {}. " \
                        "It is preceded by attributes: " \
                        "{}".format(type(layer), layer.requires_prev, 
                                    prev_layer.attributes))

def _validate_layers_parallel(layers, prev_layer, next_layer):
    """Validate that all layers have the same required features."""
    _validate_layers_layers(layers)

    # Supports prev and next layer
    for i, layer in enumerate(layers):
        if prev_layer:
            _validate_requires_next(prev_layer, layer)

        if next_layer:
            _validate_requires_prev(next_layer, layer)

def _validate_layers_sequence(layers):
    """Validate that layers are valid and each layer matches the next layer."""
    _validate_layers_layers(layers)

    # Check that feature requirements line up
    # Check each lines up with next
    for i in range(len(layers)-1):
        _validate_requires_next(layers[i], layers[i+1])

    # Check each lines up with prev
    for i in range(1, len(layers)):
        _validate_requires_prev(layers[i], layers[i-1])

def _validate_graph(graph_):
    """Validate a graph of network layers."""
    # Graph should have 'I' key
    assert 'I' in graph_.nodes

    # 'I' should have no incoming edges
    for edge in graph_.edges:
        assert edge[1] != 'I'

    # 'O' should have no outgoing edges
    for edge in graph_.edges:
        assert edge[0] != 'O'

    # Graph should have exactly one 'O' value
    O_s = 0
    for edge in graph_.edges:
        if 'O' in edge:
            O_s += 1
    assert O_s == 1

    # All nodes should be able to flow to output
    for node in graph_.nodes:
        # find path from node to 'O'
        assert graph.find_path(graph_.adjacency, node, 'O') is not None

    # Graph should be made of layers (excluding 'I' and 'O')
    _validate_layers_layers(graph_.nodes - set(['I', 'O']))

    # Validate requires next and requires prev
    for layer in graph_.nodes - set(['I', 'O']):
        # Validate requires prev
        incoming_layers = graph_.backwards_adjacency[layer]
        for incoming_layer in incoming_layers:
            if incoming_layer is not 'I':
                _validate_requires_prev(layer, incoming_layer)

        # Validate requires next
        outgoing_layers = graph_.adjacency[layer]
        for outgoing_layer in outgoing_layers:
            if outgoing_layer is not 'O':
                _validate_requires_next(layer, outgoing_layer)

###################
# Network functions
###################
def _layers_to_adjacency_dict(layers):
    """Convert sequence of layers to graph of layers."""
    # First to input
    try:
        layers_dict = {'I': [layers[0]]}
    except IndexError:
        # No layers
        return {'I': ['O']}

    # Each layer to the next
    prev_layer = layers[0]
    for layer in layers[1:]:
        layers_dict[prev_layer] = [layer]
        prev_layer = layer

    # Last to output
    layers_dict[layers[-1]] = ['O']

    return layers_dict

def _reachable_nodes_list(adjacency_dict, start):
    # Append each node to list
    expanded_nodes = []
    def node_callback(node):
        expanded_nodes.append(node)
    
    graph.traverse_bredth_first(adjacency_dict, start, node_callback)
    return expanded_nodes

def _get_all_prerequisites(graph_, node, prerequisite_node):
    # Search for a path from prerequisite node to current node in backwards_adjacency
    if graph.find_path(graph_.backwards_adjacency, prerequisite_node, node) is not None: 
        # If it exists, this is a cycle, and the prerequisite should not be added
        return []
    else:
        # Otherwise, add the node, and all of it's prerequesites (recursively)
        # before the current node.
        # TODO: fix this, since reachable nodes returns a set
        # we want a list of visited nodes, in the order visited
        reachable_nodes = _reachable_nodes_list(graph_.backwards_adjacency,
                                                prerequisite_node)

        return reversed(reachable_nodes)

def _make_activation_order(graph_):
    """Determine the order of activation for a graph of layers."""
    activation_order = []

    # For each node visited:
    def node_callback(node):
        # Check for prerequisites that need to activate first
        for prerequisite_node in graph_.backwards_adjacency[node]:
            # If a node is reached with incoming nodes not already in activation_order
            if prerequisite_node not in activation_order:
                activation_order.extend(_get_all_prerequisites(graph_,
                                                               node,
                                                               prerequisite_node))

        # Append each visited node to activation_order
        activation_order.append(node)

    # Traverse graph from 'I' to 'O' (breath first? Doesn't really matter)
    graph.traverse_bredth_first(graph_.adjacency, 'I', node_callback)

    activation_order.remove('I') # Not a layer
    activation_order.remove('O') # Not a layer
    return activation_order


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


class Network(object):
    """A composite of layers connected in sequence."""
    def __init__(self, layers, incoming_order_dict=None):
        # Allow user to pass a list of layers connected in sequence
        if isinstance(layers, list):
            layers = _layers_to_adjacency_dict(layers)

        self._graph = graph.Graph(layers)
        _validate_graph(self._graph)
        

        # User can specify an order for incoming layers.
        # This is important for some layers that have multiple inputs.
        if incoming_order_dict is not None:
            # Ensure backwards adjacency has the right order
            for layer, desired_order in incoming_order_dict.iteritems():
                current_order = self._graph.backwards_adjacency[layer]
                assert _all_in(desired_order, current_order)

                # Begin with the specified order
                new_order = desired_order[:]
                # Then add all remaining layers at the end
                for incoming_layer in current_order:
                    if incoming_layer not in new_order:
                        new_order.append(incoming_layer)

                self._graph.backwards_adjacency[layer] = new_order

        # Initialize activations
        self._activation_order = _make_activation_order(self._graph)
        self._activations = {}
        for layer in self._graph.nodes:
            self._activations[layer] = None

        # Create link from each layer to this network, so layers can
        # make changes based on other layers
        for layer in self._activation_order:
            layer.network = self

        # Bookkeeping
        self.logging = True
        self.iteration = 0

        # Initialize all layers
        self.reset()

    def _reset_bookkeeping(self):
        self.iteration = 0

    def _incoming_activations(self, layer):
        return [self._activations[incoming] for incoming in 
                self._graph.backwards_adjacency[layer]]

    def _outgoing_errors(self, layer, errors_dict):
        return [errors_dict[outgoing] for outgoing in
                self._graph.adjacency[layer]]

    def activate(self, inputs):
        """Return the network outputs for given inputs."""
        inputs = numpy.array(inputs)
        self._activations['I'] = inputs
        
        for layer in self._activation_order:
            ouputs = layer.activate(*self._incoming_activations(layer))

            # Track all activations for learning, and layer inputs
            self._activations[layer] = ouputs

        # Return activation of the only layer that feeds into output
        return self._activations[self._graph.backwards_adjacency['O'][0]]

    def update(self, inputs, targets):
        """Adjust the network towards the targets for given inputs."""
        outputs = self.activate(inputs)
        outputs_dict = {'O': outputs}

        errors = targets - outputs
        output_errors = errors # For returning
        errors_dict = {'O': errors}

        for layer in reversed(self._activation_order):
            # Grab all the variables we need from storage dicts
            all_inputs = self._incoming_activations(layer)
            all_errors = self._outgoing_errors(layer, errors_dict)
            outputs = self._activations[layer]

            # Compute errors for preceding layer before this layers changes
            errors_dict[layer] = layer.get_prev_errors(all_inputs,
                                                       all_errors, 
                                                       outputs)
                
            # Update
            layer.update(all_inputs, outputs, all_errors)

        return output_errors

    def reset(self):
        """Reset every layer in the network."""
        for layer in self._activation_order:
            layer.reset()

    def test(self, patterns):
        """Print corresponding inputs and outputs from a set of patterns."""
        for p in patterns:
            print(p[0], '->', self.activate(p[0]))

    def get_error(self, pattern):
        """Return the mean squared error (MSE) for a pattern."""
        # Mean squared error
        return numpy.mean(numpy.subtract(self.activate(pattern[0]), pattern[1])**2)

    def get_avg_error(self, patterns):
        """Return the average mean squared error for a set of patterns."""
        error = 0.0        
        for pattern in patterns:
            error = error + self.get_error(pattern)
 
        return error/len(patterns)

    def train(self, patterns, iterations=1000, error_break=0.02,
              pattern_select_func=select_iterative, post_pattern_callback=None,
              preprocess_func=None):
        """Train network to converge on set of patterns.
        
        Args:
            patterns: A set of (inputs, targets) pairs.
            iterations: Max iterations to train network.
            error_break: Training will end once error is less than this.
            pattern_select_func: Function that takes a set of patterns,
                and returns a set of patterns. Use partial function to embed arguments.
        """
        self._reset_bookkeeping()
        #self.reset()

        # For optimization
        track_error = error_break != 0.0 or self.logging

        # Preprocess data
        if preprocess_func is not None:
            patterns = preprocess_func(patterns)

        # Pre-training for each layer
        for layer in self._activation_order:
            layer.pre_training(patterns)

        # Learn on each pattern for each iteration
        for self.iteration in range(1, iterations+1):

            # Pre-iteration for each layer
            for layer in self._activation_order:
                layer.pre_iteration(patterns)

            # Learn each selected pattern
            error = 0.0
            for pattern in pattern_select_func(patterns):
                # Learn
                errors = self.update(pattern[0], pattern[1])

                # Optional callback for user extension,
                # such as a visualization or history tracking
                if post_pattern_callback:
                    post_pattern_callback(self, pattern)

                # Sum errors
                if track_error:
                    error += numpy.mean(errors**2)

            # Post-iteration for each layer
            for layer in self._activation_order:
                layer.post_iteration(patterns)
                
            # Logging and breaking
            if track_error:
                error = error / len(patterns)
                if self.logging:
                    print "Iteration {}, Error: {}".format(self.iteration, error)

                # Break early to prevent overtraining
                if error < error_break:
                    break

        # Post-training for each layer
        for layer in self._activation_order:
            layer.post_training(patterns)

    def serialize(self):
        """Convert network into string.
        
        Returns:
            string; A string representing this network.
        """
        raise NotImplementedError()

    @classmethod
    def unserialize(cls, serialized_network):
        """Convert serialized network into network.

        Returns:
            Network; A Network object.
        """
        raise NotImplementedError()

##########################
# Quick network functions
##########################
def make_mlp(shape, learn_rate=0.5, momentum_rate=0.1):
    """Create a multi-layer perceptron network."""
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
    layers.append(mlp.TanhTransferPerceptron())

    return Network(layers)

def make_dropout_mlp(shape, learn_rate=0.5, momentum_rate=0.1,
                     input_active_probability=0.8,
                     hidden_active_probability=0.5):
    """Create a multi-layer perceptron network with dropout."""
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
    # and last perceptron layer must not reduce number of outputs
    layers.append(mlp.DropoutPerceptron(shape[-2], shape[-1],
                                        learn_rate, momentum_rate,
                                        active_probability=1.0))
    layers.append(mlp.TanhTransferPerceptron())

    return Network(layers)

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

def make_pbnn():
    from pynn.architecture import pbnn
    from pynn.architecture import transfer

    # Calculate distances to stored points
    store_inputs = pbnn.StoreInputsLayer()
    distances = pbnn.DistancesLayer()

    # Gaussian transfer
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