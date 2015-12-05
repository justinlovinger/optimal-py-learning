import numpy

import network

def sigmoid(x):
    """Sigmoid like function using tanh"""
    return numpy.tanh(x)

def dsigmoid(y):
    """Derivative of sigmoid above"""
    return 1.0 - y**2

class MLP(network.Network):
    def __init__(self, shape):
        super(MLP, self).__init__()

        self.shape = shape
        self.layers = []
        self.weights = []
        self.momentums = []

        # Input layer (+1 unit for bias)
        self.layers.append(numpy.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1, len(self.shape)):
            self.layers.append(numpy.ones(self.shape[i]))

        # Build weights matrix
        for i in range(len(self.shape)-1):
            self.weights.append(numpy.zeros((self.layers[i].size,
                                             self.layers[i+1].size)))

        # momentums holds last change in weights
        self.momentums = [0,]*len(self.weights)

    def reset(self):
        # Randomize weights, between -0.25 and 0.25
        for i in range(len(self.weights)):
            random_matrix = numpy.random.random((self.layers[i].size,
                                              self.layers[i+1].size))
            self.weights[i] = (2*random_matrix-1)*0.25

    def activate(self, inputs):
        """Propagate data from input layer to output layer."""
        if len(inputs) != self.shape[0]:
            raise ValueError('wrong number of inputs')

        # Set input layer
        self.layers[0][0:-1] = inputs

        # Activate each layer, with output from last layer
        for i in range(1, len(self.shape)):
            self.layers[i] = sigmoid(numpy.dot(self.layers[i-1], self.weights[i-1]))

        # Return last output as final output
        return self.layers[-1]

    def compute_deltas(self, targets):
        deltas = []

        # Compute error on output layer
        error = targets - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = numpy.dot(deltas[0], self.weights[i].T) * dsigmoid(self.layers[i])
            deltas.insert(0, delta)

        return deltas, error

    def backpropagate(self, targets, learn_rate=0.5, momentum=0.1):
        """Back propagate error related to target."""
        if len(targets) != self.shape[-1]:
            raise ValueError('wrong number of target values')

        deltas, error = self.compute_deltas(targets)
            
        # Update weights
        for i in range(len(self.weights)):
            # Update, [:,None] quickly transposes an array to a col vector
            change = self.layers[i][:,None] * deltas[i]
            self.weights[i] += learn_rate*change + momentum*self.momentums[i]

            # Save change ans momentum for next backpropogate
            self.momentums[i] = change

        # Return error
        #return (error**2).sum()

    def learn(self, inputs, targets, learn_rate=0.5, momentum=0.1):
        self.activate(inputs)
        self.backpropagate(targets, learn_rate, momentum)

if __name__ == '__main__':
    import time
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # Create a network with two input, two hidden, and one output nodes
    n = MLP((2, 10, 1))

    # Train it with some patterns
    start = time.clock()
    n.logging = False
    n.train(pat, 1000, 0.0, learn_rate=0.5, momentum=0.1)
    print time.clock() - start
    # test it
    n.test(pat)