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

from learning import datasets, validation, MLP
from learning import SoftmaxTransfer  # To further customize our MLP
from learning import CrossEntropyError  # To customize the error function of our MLP
from learning import optimize  # To customize the training of our MLP

# Grab the popular iris dataset, from our library of datasets
dataset = datasets.get_iris()

# Make a multilayer perceptron to classify the iris dataset
model = MLP(
    # The MLP will take 4 attributes, have 1 hidden layer with 2 neurons,
    # and outputs one of 3 classes
    (4, 2, 3),

    # We will use a softmax output layer for this classification problem
    # Because we are only changing the output transfer, we pass a single
    # Transfer object. We could customize all transfer layers by passing
    # a list of Transfer objects.
    transfers=SoftmaxTransfer(),

    # Cross entropy error will pair nicely with our softmax output.
    error_func=CrossEntropyError(),

    # Lets use the quasi-newton BFGS optimizer for this problem
    # BFGS requires and n^2 operation, where n is the number of weights,
    # but this isn't a problem for our relatively small MLP.
    # If we don't want to deal with optimizers, the default
    # option will select an appropriate optimizer for us.
    optimizer=optimize.BFGS(
        # We can even customize the line search method
        step_size_getter=optimize.WolfeLineSearch(
            # And the initial step size for our line search
            initial_step_getter=optimize.FOChangeInitialStep())
    ))

# NOTE: For rapid prototyping, we could quickly implement an MLP as follows
# model = MLP((4, 2, 3))

# Lets train our MLP
# First, we'll split our dataset into training and testing sets
# Our training set will contain 30 samples from each class
training_set, testing_set = validation.make_train_test_sets(*dataset, train_per_class=30)

# We could customize training and stopping criteria through
# the arguments of train, but the defaults should be sufficient here
model.train(*training_set)

# Our MLP should converge in a couple of seconds
# Lets see how our MLP does on the testing set
print 'Testing accuracy:', validation.get_accuracy(model, *testing_set)
