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

# Add model building
from learning.base import Model

# Add datasets
from learning.data import datasets

# Add transfer functions
from learning.transfer import (LinearTransfer, TanhTransfer, ReluTransfer,
                               GaussianTransfer, SoftmaxTransfer)

# Add models
from learning.architecture.multioutputs import MultiOutputs
from learning.architecture.som import SOM
from learning.architecture.mlp import MLP, DropoutMLP
from learning.architecture.rbf import RBF
from learning.architecture.pbnn import PBNN
from learning.architecture.regression import LinearRegressionModel
