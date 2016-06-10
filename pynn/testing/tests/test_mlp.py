import random

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
    layer._weights = weights
    layer._full_weights = weights

    # Make network to test incoming weight matrix reduction
    prev_layer = mlp.DropoutPerceptron(1, 2)
    nn = network.Network(prev_layer, layer)

    monkeypatch.setattr(mlp, '_random_indexes', lambda a, b : [0])
    prev_layer.pre_iteration([])
    monkeypatch.undo()

    # pre_iteration hook should reduce incoming component of weight matrix
    # based on incoming dropout perceptrons
    layer.pre_iteration([])

    assert layer._weights == numpy.array([[0.0, 1.0]])

    # Test for second row
    monkeypatch.setattr(mlp, '_random_indexes', lambda a, b : [1])
    prev_layer.pre_iteration([])
    monkeypatch.undo()

    layer.pre_iteration([])

    assert layer._weights == numpy.array([[2.0, 3.0]])


def test_dropout_perceptron_post_iteration(monkeypatch):
    assert 0


def test_dropout_perceptron_post_training():
    assert 0


def test_random_indexes_probability_one():
    length = random.randint(1, 10)
    assert mlp._random_indexes(length, 1.0) == range(length)
