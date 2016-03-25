from pynn import network

class SetOutputLayer(network.Layer):
    num_inputs = 'any'

    def __init__(self, output):
        super(SetOutputLayer, self).__init__()

        self.output = output
        self.num_outputs = len(output)

    def activate(self, inputs):
        return self.output

    def reset(self):
        pass

    def update(self, inputs, outputs, errors):
        pass