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

import numpy

from learning import Model


class Ensemble(Model):
    def __init__(self, networks):
        super(Ensemble, self).__init__()

        self._networks = networks
        self.reset()


class Bagger(Ensemble):
    requires_prev = (None, )

    def reset(self):
        super(Bagger, self).reset()

        for network in self._networks:
            network.reset()

    def activate(self, inputs):
        # Unweighted average of layer outputs
        output = numpy.array(self._networks[0].activate(inputs), dtype='d')
        for network in self._networks[1:]:
            output += network.activate(inputs)

        return output / len(self._networks)

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        return None  #TODO

    def update(self, inputs, outputs, errors):
        for network in self._networks:
            network.update(inputs, errors + outputs)
