import numpy

class Layer(object):
    def activate(self, inputs):
        raise NotImplementedError()

    def get_deltas(self, errors, outputs):
        raise NotImplementedError()

    def get_errors(self, deltas):
        raise NotImplementedError()

    def update(self, inputs, deltas):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

class GrowingLayer(Layer):
    """Layer that can add new neurons, increasing output size.
    
    Raise growth exception when new neuron is added.
    """
    pass

class SupportsGrowingLayer(Layer):
    """Layer that supports new neurons by increasing input size.
    
    Adds neurons when GrowingLayer before it raises growth exception.
    """
    def add_input(self):
        raise NotImplementedError()

class Ensemble(Layer):
    def __init__(self, layers):
        super(Ensemble, self).__init__()

        self._layers = layers
        self.reset()

class Network(object):
    def __init__(self, layers):
        # Assert each element of layer is a Layer,
        # and layers properly line up (inputs -> outputs, Growing -> SupportsGrowing)
        # Otherwise, raise value error
        # TODO

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

    def learn(self, inputs, targets):
        output = self.activate(inputs)

        errors = targets - output
        output_errors = errors # for returning

        for i, layer in enumerate(reversed(self._layers)):
            # Pseudo reverse activations, so they are in the right order for
            # reversed layers list
            layer_inputs = self._activations[len(self._layers)-i-1]
            layer_outputs = self._activations[len(self._layers)-i]

            # Compute deltas for this layer update
            deltas = layer.get_deltas(errors, layer_outputs)

            # Compute errors for next layer deltas
            errors = layer.get_errors(deltas, layer_outputs)

            # Update
            layer.update(layer_inputs, deltas)

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