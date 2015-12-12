import numpy

class Layer(object):
    """A layer of computation for a supervised learning network."""
    attributes = tuple([])

    requires_prev = tuple([])
    requires_next = tuple([])
    
    num_inputs = None
    num_outputs = None

    def reset(self):
        raise NotImplementedError()

    def activate(self, inputs):
        raise NotImplementedError()

    def get_prev_errors(self, errors, outputs):
        raise NotImplementedError()

    def update(self, inputs, outputs, errors):
        raise NotImplementedError()

    def get_outputs(self, inputs, outputs):
        """Get outputs for previous layer.
        
        Transfer functions should override to properly transform their input
        """
        return inputs

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
    pass

class Ensemble(Layer):
    """A composite of layers connected in parallel."""
    def __init__(self, layers):
        super(Ensemble, self).__init__()

        self._layers = layers
        self.reset()

def _validate_layers(layers):
    """Validate that each layer matches the next layer."""
    # Assert each element of layers is a Layer
    for layer in layers:
        if not isinstance(layer, Layer):
            raise ValueError("layers argument of Network must contain Layer's." \
                             " Instead contains {}.".format(type(layer)))

    # And that inputs --> outputs line up
    last_num = 'any'
    for i in range(len(layers)-1):
        layer = layers[i]
        next_layer = layers[i+1]

        # Track how many inputs are required for next layer
        if layer.num_outputs == '+1': # Such as bias
            if last_num == 'any':
                continue
            else:
                last_num = last_num + 1
        elif layer.num_outputs != 'any': # When any, last_num doesn't change
            last_num = layer.num_outputs

        if next_layer.num_inputs == 'any':
            continue

        # Validate
        if last_num != 'any' and next_layer.num_inputs != last_num:
            raise ValueError('num_inputs for layer {} does not match' \
                             ' preceding layers'.format(i+2)) # Starts at 1

    # Check that feature requirements line up
    # Check each lines up with next
    for i in range(len(layers)-1):
        for attribute in layers[i].requires_next:
            if not attribute in layers[i+1].attributes:
                raise TypeError("Layer of type {} must be followed by attributes: {}. " \
                                "It is followed by attributes: " \
                                "{}".format(type(layers[i]), layers[i].requires_next, 
                                            layers[i+1].attributes))
    # Check each lines up with prev
    for i in range(1, len(layers)):
        for attribute in layers[i].requires_prev:
            if not attribute in layers[i-1].attributes:
                raise TypeError("Layer of type {} must be preceded by attributes: {}. " \
                                "It is preceded by attributes: " \
                                "{}".format(type(layers[i]), layers[i].requires_prev, 
                                            layers[i-1].attributes))

def _find_num_inputs(layers):
    """Determine number of initial inputs for a series of layers."""
    offset = 0

    for layer in layers:
        if layer.num_inputs != 'any':
            return layer.num_inputs - offset

        if layer.num_outputs == '+1':
            offset += 1

    return 'any'

def _find_num_outputs(layers):
    """Determine final number of outputs for a series of layers."""
    for layer in reversed(layers):
        if layer.num_outputs != 'any':
            return layer.num_outputs

    return 'any'

class Network(object):
    """A composite of layers connected in series."""
    def __init__(self, layers):
        _validate_layers(layers)

        self._layers = layers
        self._num_inputs = _find_num_inputs(layers)
        self._num_outputs = _find_num_outputs(layers)
        self._activations = []

        self.logging = True

        # Bookkeeping
        self.iteration = 0

    def _reset_bookkeeping(self):
        self.iteration = 0

    def activate(self, inputs):
        if self._num_inputs != 'any' and len(inputs) != self._num_inputs:
            raise ValueError('Wrong number of inputs. Expected {}, got {}' \
                             ''.format(self._num_inputs, len(inputs)))

        inputs = numpy.array(inputs)
        self._activations = [inputs]

        for layer in self._layers:
            inputs = layer.activate(inputs)

            # Track all activations for learning
            self._activations.append(inputs)

        return inputs

    def learn(self, first_inputs, targets):
        if self._num_outputs != 'any' and len(targets) != self._num_outputs:
            raise ValueError('Wrong number of targets. Expected {}, got {}' \
                             ''.format(self._num_outputs, len(targets)))

        outputs = self.activate(first_inputs)

        errors = targets - outputs
        output_errors = errors # for returning

        for i, layer in enumerate(reversed(self._layers)):
            # Pseudo reverse activations, so they are in the right order for
            # reversed layers list
            inputs = self._activations[len(self._layers)-i-1]

            # Compute errors for preceding layer before this layers changes
            prev_errors = layer.get_prev_errors(errors, outputs)

            # Update
            layer.update(inputs, outputs, errors)

            # Setup for preceding layer
            errors = prev_errors
            # Outputs for preceding layer are this layers inputs
            outputs = layer.get_outputs(inputs, outputs)

        return output_errors

    def reset(self):
        for layer in self._layers:
            layer.reset()

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.activate(p[0]))

    def get_single_error(self, output, targets):
        error = 0.0       
        for k in range(len(targets)):
            error = error + (targets[k]-output[k])**2
        return error
        
    def get_error(self, patterns):
        error = 0.0        
        for pattern in patterns:
            inputs  = pattern[0]
            targets = pattern[1]
            output = self.activate(inputs)
            error = error + self.get_single_error(output, targets)  
        error = error/len(patterns)         
        return error

    def random_train(self, patterns, N, M):
        error = 0.0
        for i in range(len(patterns)):
            p = patterns[random.randint(0, len(patterns)-1)]
            inputs = p[0]
            targets = p[1]
            self.update(inputs)
            error = error + self.backPropagate(targets, N, M)
        return error
        
    def iterative_train(self, patterns, N, M):
        error = 0.0
        #random.shuffle(pattern_indexes)
        #for index in pattern_indexes:
            #p = patterns[index]
        for p in patterns:
            inputs = p[0]
            targets = p[1]
            self.update(inputs)
            error = error + self.backPropagate(targets, N, M)
        return error
        
    def shuffle_train(self, patterns, N, M):
        error = 0.0
        pattern_indexes = range(len(patterns))
        pattern_indexes = pattern_indexes
        random.shuffle(pattern_indexes)
        for index in pattern_indexes:
            p = patterns[index]
            inputs = p[0]
            targets = p[1]
            self.update(inputs)
            error = error + self.backPropagate(targets, N, M)
        return error
    
    def train(self, patterns, iterations=1000, error_break=0.02, *args, **kwargs):
        self._reset_bookkeeping()
        self.reset()

        # For optimization
        track_error = error_break != 0.0 or self.logging

        # Learn on each pattern for each iteration
        for self.iteration in range(iterations):
            error = 0.0       
            for pattern in patterns:
                # Learn
                errors = self.learn(pattern[0], pattern[1], *args, **kwargs)

                # Sum errors
                if track_error:
                    error += sum(errors**2)
                
            if track_error:
                error = error / len(patterns)
                if self.logging:
                    print "Iteration {}, Error: {}".format(self.iteration, error)

                # Break early to prevent overtraining
                if error < error_break:
                    return

def make_mlp(shape, learn_rate=0.5, momentum_rate=0.1):
    """Create a multi-layer perceptron network."""
    from pynn import transfer
    from pynn import transform

    layers = []
    # Create first layer with bias
    layers.append(transform.AddBias())
    layers.append(transform.Perceptron(shape[0]+1, shape[1], False, 
                                       learn_rate, momentum_rate))
    layers.append(transfer.SigmoidTransfer())

    # Create other layers without bias
    for i in range(1, len(shape)-1):
        layers.append(transform.Perceptron(shape[i], shape[i+1], False, 
                                           learn_rate, momentum_rate))
        layers.append(transfer.SigmoidTransfer())

    return Network(layers)

def make_rbf(inputs, neurons, outputs, learn_rate=1.0,
             move_rate=0.1, neighborhood=2, neighbor_move_rate=1.0):
    """Create a radial-basis function network."""
    from pynn import transfer
    from pynn import transform
    from pynn import som

    layers = [
              som.SOM(inputs, neurons, move_rate, neighborhood, neighbor_move_rate),
              transfer.GaussianTransfer(),
              transform.GaussianOutput(neurons, outputs, learn_rate),
             ]

    return Network(layers)