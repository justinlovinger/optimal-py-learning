import random
import copy

import pytest
import numpy

from learning.architecture import mlp
from learning.data import datasets
from learning import base

from learning.testing import helpers

############################
# MLP
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
    nn = mlp.MLP((2, 4, 2))
    pat = datasets.get_xor()

    nn.train(*pat, retries=5, error_break=0.002)
    assert nn.avg_mse(*pat) <= 0.02


def test_mlp_classifier():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = mlp.MLP((2, 2, 2), transfers=mlp.SoftmaxTransfer())
    pat = datasets.get_xor()

    error = nn.avg_mse(*pat)
    nn.train(*pat, iterations=10)
    assert nn.avg_mse(*pat) < error


pytest.mark.slowtest()
def test_mlp_classifier_convergence():
    # Run until convergence
    # assert that network can converge
    nn = mlp.MLP((2, 3, 2), transfers=mlp.SoftmaxTransfer())
    pat = datasets.get_and()

    nn.train(*pat, retries=5, error_break=0.002)
    assert nn.avg_mse(*pat) <= 0.02


def test_mlp_bias():
    # Should have bias for each layer
    model = mlp.MLP((2, 4, 3))

    # +1 input for bias
    assert model._weight_matrices[0].shape == (3, 4)
    assert model._weight_matrices[1].shape == (5, 3)

    # First input should always be 1
    model.activate([0, 0])
    assert model._weight_inputs[0][0] == 1.0
    assert model._weight_inputs[1][0] == 1.0
    assert model._weight_inputs[2][0] == 1.0

def test_mlp_perceptron():
    # Given known inputs and weights, test expected outputs
    model = mlp.MLP((2, 1), transfers=mlp.LinearTransfer())
    model._weight_matrices[0][0][0] = 0.0
    model._weight_matrices[0][1][0] = 0.5
    model._weight_matrices[0][2][0] = -0.5
    assert (model.activate([1, 1]) == [0.0]).all()

    model._weight_matrices[0][1][0] = 1.0
    model._weight_matrices[0][2][0] = 2.0
    assert (model.activate([1, 1]) == [3.0]).all()

##############################
# DropoutMLP
##############################
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
    nn = mlp.DropoutMLP((2, 6, 3, 2),
                        input_active_probability=1.0,
                        hidden_active_probability=0.9)
    pat = datasets.get_and() # Easier and dataset for lienar output

    # Error break lower than cutoff, since dropout may have different error
    # after training
    nn.train(*pat, retries=5, error_break=0.002, error_improve_iters=50)

    # Dropout sacrifices training accuracy for better generalization
    # so we don't worry as much about convergence
    assert nn.avg_mse(*pat) <= 0.1


def test_dropout_mlp_classifier():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = mlp.DropoutMLP((2, 6, 3, 2), transfers=mlp.SoftmaxTransfer())
    pat = datasets.get_and()

    error = nn.avg_mse(*pat)
    nn.train(*pat, iterations=10)
    assert nn.avg_mse(*pat) < error


pytest.mark.slowtest()
def test_dropout_mlp_classifier_convergence():
    # Run until convergence
    # assert that network can converge
    # Since XOR does not really need dropout, we use high probabilities
    nn = mlp.DropoutMLP((2, 6, 3, 2), transfers=mlp.SoftmaxTransfer(),
                        input_active_probability=1.0,
                        hidden_active_probability=0.9)
    pat = datasets.get_and()

    # Error break lower than cutoff, since dropout may have different error
    # after training
    nn.train(*pat, retries=5, error_break=0.002, error_improve_iters=50)

    # Dropout sacrifices training accuracy for better generalization
    # so we don't worry as much about convergence
    assert nn.avg_mse(*pat) <= 0.1

def test_dropout_mlp_dropout():
    model = mlp.DropoutMLP((2, 4, 3), input_active_probability=0.5,
                           hidden_active_probability=0.5)

    # Only bias and active neurons should not be 0
    model.train_step([[1, 1]], [[1, 1, 1]])

    # Should still have DropoutTransfers (until activation outside of training)

    _validate_weight_inputs(model._weight_inputs[0], model._input_transfer._active_neurons)
    for weight_inputs, transfer_func in zip(model._weight_inputs[1:-1], model._transfers[:-1]):
        _validate_weight_inputs(weight_inputs, transfer_func._active_neurons)

def test_dropout_mlp_post_training():
    # Post training should happen on first activate after training (train step),
    # and not more than once, unless training begins again
    model = mlp.DropoutMLP((2, 4, 3), input_active_probability=0.5,
                           hidden_active_probability=0.5)

    # Train, modifying active neurons and weights
    model.train_step([[1, 1]], [[1, 1, 1]])
    pre_procedure_weights = copy.deepcopy(model._weight_matrices)

    # Should call post_training procedure after activate
    model.activate([1, 1])
    _validate_post_training(model, pre_procedure_weights)

    # Weights should not change after another activate
    model.activate([1, 1])
    _validate_post_training(model, pre_procedure_weights)

def _validate_post_training(model, pre_procedure_weights):
    for weight_matrix, orig_matrix in zip(model._weight_matrices, pre_procedure_weights):
        # Weights should be scaled by active probability
        assert (weight_matrix == orig_matrix*model._hid_act_prob).all()

    # All inputs and neurons should be active
    assert not isinstance(model._input_transfer, mlp.DropoutTransfer)
    for transfer_func in model._transfers:
        assert not isinstance(transfer_func, mlp.DropoutTransfer)

    for weight_inputs in model._weight_inputs:
        _validate_weight_inputs(weight_inputs, [1.0]*(len(weight_inputs)-1))


def _validate_weight_inputs(weight_inputs, active_neurons):
    assert len(weight_inputs)-1 == len(active_neurons) # -1 for bias

    assert weight_inputs[0] == 1.0 # Bias
    for i, active in enumerate(active_neurons):
        # i+1 to offset from bias
        if active == 0.0:
            assert weight_inputs[i+1] == 0.0
        elif active == 1.0:
            assert weight_inputs[i+1] != 0.0
        else:
            assert 0, 'Invalid active neuron value'

####################
# DropoutTransfer
####################
def test_dropout_transfer_probability_one():
    length = random.randint(1, 20)

    dropout_transfer = mlp.DropoutTransfer(mlp.LinearTransfer(), 1.0, length)
    assert (dropout_transfer._active_neurons == numpy.array([1.0]*length)).all(), 'All should be active'

    # Random input
    input_vec = numpy.random.random(length)
    assert (dropout_transfer(input_vec) == input_vec).all()

def test_dropout_transfer_probability_zero():
    length = random.randint(1, 20)

    # Can't actually be zero, but can be close enough
    dropout_transfer = mlp.DropoutTransfer(mlp.LinearTransfer(), 1e-16, length)

    # Should not allow zero active, defaults to 1
    assert list(dropout_transfer._active_neurons).count(1.0) == 1
    assert list(dropout_transfer._active_neurons).count(0.0) == length-1
