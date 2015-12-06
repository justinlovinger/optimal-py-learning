import numpy

class Layer(object):
    pass

class Layer(Layer):
    requires_prev = (Layer,)
    requires_next = (Layer,)
    
    def reset(self):
        raise NotImplementedError()

    def activate(self, inputs):
        raise NotImplementedError()

    def get_deltas(self, errors, outputs):
        raise NotImplementedError()

    def get_errors(self, deltas, outputs):
        raise NotImplementedError()

    def update(self, inputs, deltas):
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
    def add_input(self):
        raise NotImplementedError()

class GrowingLayer(Layer):
    """Layer that can add new neurons, increasing output size.
    
    Raise growth exception when new neuron is added.
    """

    requires_next = (SupportsGrowingLayer,)
    pass

class Ensemble(Layer):
    def __init__(self, layers):
        super(Ensemble, self).__init__()

        self._layers = layers
        self.reset()

class Network(object):
    def __init__(self, layers):
        # Assert each element of layers is a Layer
        # And that inputs --> outputs line up, TODO
        for layer in layers:
            if not isinstance(layer, Layer):
                raise ValueError("layers argument of Network must contain Layer's."
                                 " Instead contains {}.".format(type(layer)))

        # Check each lines up with next
        for i in range(len(layers)-1):
            if not isinstance(layers[i+1], layers[i].requires_next):
                raise ValueError("Layer of type {} must be followed by type: {}. " \
                                 "It is followed by type: " \
                                 "{}".format(type(layers[i]), layers[i].requires_next, 
                                             type(layers[i+1])))
        # Check each lines up with prev
        for i in range(1, len(layers)):
            if not isinstance(layers[i-1], layers[i].requires_next):
                raise ValueError("Layer of type {} must be preceded by type: {}. " \
                                 "It is preceded by type: " \
                                 "{}".format(type(layers[i]), layers[i].requires_next, 
                                             type(layers[i-1])))


        self._layers = layers
        self._activations = []

        self.logging = True

        # Bookkeeping
        self.iteration = 0

    def _reset_bookkeeping(self):
        self.iteration = 0

    def activate(self, inputs):
        inputs = numpy.array(inputs)
        self._activations = [inputs]

        for layer in self._layers:
            inputs = layer.activate(inputs)

            # Track all activations for learning
            self._activations.append(inputs)

        return inputs

    def learn(self, first_inputs, targets):
        outputs = self.activate(first_inputs)

        errors = targets - outputs
        output_errors = errors # for returning

        for i, layer in enumerate(reversed(self._layers)):
            # Pseudo reverse activations, so they are in the right order for
            # reversed layers list
            inputs = self._activations[len(self._layers)-i-1]

            # Compute deltas for this layer update
            deltas = layer.get_deltas(errors, outputs)

            # Compute errors for next layer deltas
            errors = layer.get_errors(deltas, outputs)

            # Update
            layer.update(inputs, deltas)

            # Outputs for next layer are this layers inputs
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