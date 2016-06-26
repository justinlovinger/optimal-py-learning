import random
import copy

import numpy

from pynn.architecture import mlp
from pynn import network

from pynn.testing import helpers

def test_perceptron():
    # Given known inputs, test expected outputs
    layer = mlp.Perceptron(2, 1)
    layer._weights[0][0] = 0.5
    layer._weights[1][0] = -0.5
    assert layer.activate(numpy.array([1, 1])) == 0.0

    layer._weights[0][0] = 1.0
    layer._weights[1][0] = 2.0
    assert layer.activate(numpy.array([1, 1])) == 3.0

def test_add_bias():
    # Given 0 vector inputs, output should not be 0
    bias = mlp.AddBias(mlp.Perceptron(2, 1))
    assert bias.activate(numpy.array([0]))[0] != 0.0

    # Without bias, output should be 0
    layer = mlp.Perceptron(1, 1)
    assert layer.activate(numpy.array([0]))[0] == 0.0


####################
# DropoutPerceptron
####################
# pre_iteration
def test_dropout_perceptron_pre_iteration_reduce_outgoing(monkeypatch):
    # Set weights for later comparison
    weights = numpy.array([[0.0, 1.0],
                           [2.0, 3.0]])

    layer = mlp.DropoutPerceptron(2, 2)
    layer._weights = weights
    layer._full_weights = weights

    # pre_iteration hook should reduce weight matrix for outgoing weights
    monkeypatch.setattr(mlp, '_random_indexes', lambda a, b : [0])
    layer.pre_iteration([])

    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[0.0],
                                                     [2.0]])))

    # Test for second column
    monkeypatch.setattr(mlp, '_random_indexes', lambda a, b : [1])
    layer.pre_iteration([])

    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[1.0],
                                                     [3.0]])))


def test_dropout_perceptron_pre_iteration_reduce_incoming(monkeypatch):
    # Set weights for later comparison
    weights = numpy.array([[0.0, 1.0],
                           [2.0, 3.0]])

    # Should not deactivate, so we only test incoming
    layer = mlp.DropoutPerceptron(2, 2, active_probability=1.0)

    # Make network to test incoming weight matrix reduction
    prev_layer = mlp.DropoutPerceptron(1, 2)
    nn = network.Network([prev_layer, layer])

    layer._weights = weights
    layer._full_weights = weights

    # pre_iteration hook should reduce incoming component of weight matrix
    # based on incoming dropout perceptrons
    prev_layer._active_neurons = [0]
    layer.pre_iteration([])

    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[0.0, 1.0]])))

    # Test for second row
    prev_layer._active_neurons = [1]
    layer.pre_iteration([])

    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[2.0, 3.0]])))


def test_dropout_perceptron_pre_iteration_correct_order(monkeypatch):
    # Set weights for later comparison
    weights = numpy.array([[0.0, 1.0],
                           [2.0, 3.0]])

    # Create network with two dropout layers
    layer = mlp.DropoutPerceptron(2, 2)
    prev_layer = mlp.DropoutPerceptron(1, 2)
    nn = network.Network([prev_layer, layer])

    layer._weights = weights
    layer._full_weights = weights

    # Disable other training functions
    monkeypatch.setattr(mlp.DropoutPerceptron, 'update', lambda *args : None)
    monkeypatch.setattr(mlp.DropoutPerceptron, 'post_iteration', lambda *args : None)
    monkeypatch.setattr(mlp.DropoutPerceptron, 'post_training', lambda *args : None)
    
    # prev_layer should set active neurons first, such that will adjust
    # based on incoming active neurons
    monkeypatch.setattr(mlp, '_random_indexes', lambda *args : [0])
    nn.train([[0.0, 0.0], [0.0, 0.0]], 1)

    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[0.0]])))


# post_iteration
def test_dropout_perceptron_post_iteration(monkeypatch):
    layer = mlp.DropoutPerceptron(2, 2)
    prev_layer = mlp.DropoutPerceptron(1, 2)
    nn = network.Network([prev_layer, layer])

    layer._full_weights = numpy.array([[-1.0, -2.0],
                                       [-3.0, -4.0]])

    # Pretend specific neurons are activated
    prev_layer._active_neurons = [0]
    layer._active_neurons = [0]

    # And weights are updated
    layer._weights = numpy.array([[1.0]])

    # post_iteration callback should update full_weights, but only those
    # for active neurons
    layer.post_iteration([])
    assert (helpers.sane_equality_array(layer._full_weights) ==
            helpers.sane_equality_array(numpy.array([[1.0, -2.0],
                                                     [-3.0, -4.0]])))

    # Try again with different active neurons. Both updates should take effect.
    prev_layer._active_neurons = [0, 1]
    layer._active_neurons = [1]

    layer._weights = numpy.array([[2.0],
                                  [4.0]])

    layer.post_iteration([])
    assert (helpers.sane_equality_array(layer._full_weights) ==
            helpers.sane_equality_array(numpy.array([[1.0, 2.0],
                                                     [-3.0, 4.0]])))

    # This time, all are active
    prev_layer._active_neurons = [0, 1]
    layer._active_neurons = [0, 1]

    layer._weights = numpy.array([[5.0, 6.0],
                                  [7.0, 8.0]])

    layer.post_iteration([])
    assert (helpers.sane_equality_array(layer._full_weights) ==
            helpers.sane_equality_array(numpy.array([[5.0, 6.0],
                                                     [7.0, 8.0]])))

# post_training
def test_dropout_perceptron_post_training():
    layer = mlp.DropoutPerceptron(2, 2, active_probability=0.5)
    layer._full_weights = numpy.array([[0.0, 1.0],
                                       [2.0, 3.0]])

    # post_training hook activates all neurons, and
    # scales weights them based on active_probability
    layer.post_training([])

    assert layer._active_neurons == [0, 1]
    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[0.0, 0.5],
                                                     [1.0, 1.5]])))


####################
# DropoutInputs
####################
def test_dropout_inputs_activate_adds_bias():
    layer = mlp.DropoutInputs(2)
    assert (helpers.sane_equality_array(layer.activate(numpy.array([0.0, 0.0]))) ==
            helpers.sane_equality_array(numpy.array([0.0, 0.0, 1.0])))

def test_dropout_inputs_activate_inputs_disabled_bias():
    layer = mlp.DropoutInputs(2)

    layer._active_neurons = [0, 2]
    assert (helpers.sane_equality_array(layer.activate(numpy.array([0.1, 0.2]))) ==
            helpers.sane_equality_array(numpy.array([0.1, 1.0])))

    layer._active_neurons = [1, 2]
    assert (helpers.sane_equality_array(layer.activate(numpy.array([0.1, 0.2]))) ==
            helpers.sane_equality_array(numpy.array([0.2, 1.0])))

def test_dropout_inputs_pre_training_disables_inputs_not_bias(monkeypatch):
    layer = mlp.DropoutInputs(2)

    monkeypatch.setattr(mlp, '_random_indexes', lambda *args : [0])
    layer.pre_iteration([])
    assert layer._active_neurons == [0, 2]

    monkeypatch.setattr(mlp, '_random_indexes', lambda *args : [1])
    layer.pre_iteration([])
    assert layer._active_neurons == [1, 2]

def test_dropout_inputs_post_training_all_active():
    layer = mlp.DropoutInputs(2)
    layer._active_neurons = [0, 2]

    layer.post_training([])
    assert layer._active_neurons == [0, 1, 2]

####################
# random_indexes
####################
def test_random_indexes_probability_one():
    length = random.randint(1, 10)
    assert mlp._random_indexes(length, 1.0) == range(length)


def test_random_indexes_probability_zero():
    length = random.randint(1, 10)

    # Will always select at least 1
    selected_indexes = mlp._random_indexes(length, 0.0)
    assert len(selected_indexes) == 1
    assert 0 <= selected_indexes[0] < length # In range