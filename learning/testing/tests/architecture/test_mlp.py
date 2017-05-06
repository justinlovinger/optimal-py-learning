import random
import copy

import pytest
import numpy

from learning.architecture import mlp
from learning.data import datasets
from learning import base

from learning.testing import helpers

############################
# Integration tests
############################
# TODO: use validation methods to more robustly test
def test_mlp():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = mlp.MLP((2, 2, 2))
    pat = datasets.get_xor()

    error = nn.avg_mse(*pat)
    nn.train(*pat, iterations=10)
    assert nn.avg_mse(*pat) < error


pytest.mark.slowtest()
def test_mlp_convergence():
    # Run until convergence
    # assert that network can converge
    nn = mlp.MLP((2, 4, 2), learn_rate=0.05, momentum_rate=0.5)
    pat = datasets.get_xor()

    nn.train(*pat, retries=5, error_break=0.002)
    assert nn.avg_mse(*pat) <= 0.02


def test_mlp_classifier():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = mlp.MLP((2, 2, 2), transfers=mlp.SoftmaxTransferPerceptron())
    pat = datasets.get_xor()

    error = nn.avg_mse(*pat)
    nn.train(*pat, iterations=10)
    assert nn.avg_mse(*pat) < error


pytest.mark.slowtest()
def test_mlp_classifier_convergence():
    # Run until convergence
    # assert that network can converge
    nn = mlp.MLP((2, 3, 2), transfers=mlp.SoftmaxTransferPerceptron(),
                 learn_rate=0.05, momentum_rate=0.5)
    pat = datasets.get_and()

    nn.train(*pat, retries=5, error_break=0.002)
    assert nn.avg_mse(*pat) <= 0.02


def test_dropout_mlp():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = mlp.DropoutMLP((2, 2, 2))
    pat = datasets.get_xor()

    error = nn.avg_mse(*pat)
    nn.train(*pat, iterations=10)
    assert nn.avg_mse(*pat) < error


pytest.mark.slowtest()
def test_dropout_mlp_convergence():
    # Run until convergence
    # assert that network can converge
    # Since XOR does not really need dropout, we use high probabilities
    nn = mlp.DropoutMLP((2, 6, 3, 2), learn_rate=0.1, momentum_rate=0.5,
                        input_active_probability=1.0,
                        hidden_active_probability=0.9)
    pat = datasets.get_and() # Easier and dataset for lienar output

    # Error break lower than cutoff, since dropout may have different error
    # after training
    nn.train(*pat, retries=5, error_break=0.002, pattern_select_func=base.select_sample)

    # Dropout sacrifices training accuracy for better generalization
    # so we don't worry as much about convergence
    assert nn.avg_mse(*pat) <= 0.1


def test_dropout_mlp_classifier():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = mlp.DropoutMLP((2, 6, 3, 2), transfers=mlp.SoftmaxTransferPerceptron(),
                        learn_rate=0.2, momentum_rate=0.5)
    pat = datasets.get_and()

    error = nn.avg_mse(*pat)
    nn.train(*pat, iterations=10, pattern_select_func=base.select_sample)
    assert nn.avg_mse(*pat) < error


pytest.mark.slowtest()
def test_dropout_mlp_classifier_convergence():
    # Run until convergence
    # assert that network can converge
    # Since XOR does not really need dropout, we use high probabilities
    nn = mlp.DropoutMLP((2, 6, 3, 2), transfers=mlp.SoftmaxTransferPerceptron(),
                        learn_rate=0.2, momentum_rate=0.5,
                        input_active_probability=1.0,
                        hidden_active_probability=0.9)
    pat = datasets.get_and()

    # Error break lower than cutoff, since dropout may have different error
    # after training
    nn.train(*pat, retries=5, error_break=0.002)

    # Dropout sacrifices training accuracy for better generalization
    # so we don't worry as much about convergence
    assert nn.avg_mse(*pat) <= 0.1

############################
# Perceptron
############################
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
class MockDropoutLayer(object):
    """Mock dropout layer that has _active_neurons attribute with all active."""
    def __init__(self, num_outputs):
        self._active_neurons = range(num_outputs)

# pre_iteration
def test_dropout_perceptron_pre_iteration_reduce_outgoing(monkeypatch):
    # Set weights for later comparison
    weights = numpy.array([[0.0, 1.0],
                           [2.0, 3.0]])

    layer = mlp.DropoutPerceptron(2, 2, MockDropoutLayer(2))
    layer._weights = weights
    layer._full_weights = weights

    # pre_iteration hook should reduce weight matrix for outgoing weights
    monkeypatch.setattr(mlp, '_random_indexes', lambda a, b : [0])
    layer.pre_iteration([], [])

    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[0.0],
                                                     [2.0]])))

    # Test for second column
    monkeypatch.setattr(mlp, '_random_indexes', lambda a, b : [1])
    layer.pre_iteration([], [])

    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[1.0],
                                                     [3.0]])))


def test_dropout_perceptron_pre_iteration_reduce_incoming(monkeypatch):
    # Set weights for later comparison
    weights = numpy.array([[0.0, 1.0],
                           [2.0, 3.0]])

    # Make network to test incoming weight matrix reduction
    nn = mlp.DropoutMLP((1, 2, 2))
    prev_layer = nn._layers[1]
    layer = nn._layers[3]

    layer._weights = weights
    layer._full_weights = weights

    # pre_iteration hook should reduce incoming component of weight matrix
    # based on incoming dropout perceptrons
    prev_layer._active_neurons = [0]
    layer.pre_iteration([], [])

    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[0.0, 1.0]])))

    # Test for second row
    prev_layer._active_neurons = [1]
    layer.pre_iteration([], [])

    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[2.0, 3.0]])))


def test_dropout_perceptron_pre_iteration_correct_order(monkeypatch):
    # Set weights for later comparison
    weights = numpy.array([[0.0, 1.0],
                           [2.0, 3.0]])

    # Create network with two dropout layers
    nn = mlp.DropoutMLP((1, 2, 2, 1))
    prev_layer = nn._layers[1]
    layer = nn._layers[3]

    layer._weights = weights
    layer._full_weights = weights

    # Disable other training functions
    monkeypatch.setattr(mlp.DropoutPerceptron, 'update', lambda *args : None)
    monkeypatch.setattr(mlp.DropoutPerceptron, 'post_iteration', lambda *args : None)
    monkeypatch.setattr(mlp.DropoutPerceptron, 'post_training', lambda *args : None)

    # prev_layer should set active neurons first, such that will adjust
    # based on incoming active neurons
    monkeypatch.setattr(mlp, '_random_indexes', lambda *args : [0])
    nn.train([[0.0]], [[0.0]], 1)

    assert (helpers.sane_equality_array(layer._weights) ==
            helpers.sane_equality_array(numpy.array([[0.0]])))


# post_iteration
def test_dropout_perceptron_post_iteration(monkeypatch):
    nn = mlp.DropoutMLP((1, 2, 2))
    prev_layer = nn._layers[1]
    layer = nn._layers[3]

    layer._full_weights = numpy.array([[-1.0, -2.0],
                                       [-3.0, -4.0]])

    # Pretend specific neurons are activated
    prev_layer._active_neurons = [0]
    layer._active_neurons = [0]

    # And weights are updated
    layer._weights = numpy.array([[1.0]])

    # post_iteration callback should update full_weights, but only those
    # for active neurons
    layer.post_iteration([], [])
    assert (helpers.sane_equality_array(layer._full_weights) ==
            helpers.sane_equality_array(numpy.array([[1.0, -2.0],
                                                     [-3.0, -4.0]])))

    # Try again with different active neurons. Both updates should take effect.
    prev_layer._active_neurons = [0, 1]
    layer._active_neurons = [1]

    layer._weights = numpy.array([[2.0],
                                  [4.0]])

    layer.post_iteration([], [])
    assert (helpers.sane_equality_array(layer._full_weights) ==
            helpers.sane_equality_array(numpy.array([[1.0, 2.0],
                                                     [-3.0, 4.0]])))

    # This time, all are active
    prev_layer._active_neurons = [0, 1]
    layer._active_neurons = [0, 1]

    layer._weights = numpy.array([[5.0, 6.0],
                                  [7.0, 8.0]])

    layer.post_iteration([], [])
    assert (helpers.sane_equality_array(layer._full_weights) ==
            helpers.sane_equality_array(numpy.array([[5.0, 6.0],
                                                     [7.0, 8.0]])))

# post_training
def test_dropout_perceptron_post_training():
    layer = mlp.DropoutPerceptron(2, 2, MockDropoutLayer(2),
                                  active_probability=0.5)
    layer._full_weights = numpy.array([[0.0, 1.0],
                                       [2.0, 3.0]])

    # post_training hook activates all neurons, and
    # scales weights them based on active_probability
    layer.post_training([], [])

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
    layer.pre_iteration([], [])
    assert layer._active_neurons == [0, 2]

    monkeypatch.setattr(mlp, '_random_indexes', lambda *args : [1])
    layer.pre_iteration([], [])
    assert layer._active_neurons == [1, 2]

def test_dropout_inputs_post_training_all_active():
    layer = mlp.DropoutInputs(2)
    layer._active_neurons = [0, 2]

    layer.post_training([], [])
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