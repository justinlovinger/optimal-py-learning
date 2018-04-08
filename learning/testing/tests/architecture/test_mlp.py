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

import random
import copy

import pytest
import numpy

from learning import (datasets, validation, LinearTransfer, SoftmaxTransfer, MeanSquaredError,
                      CrossEntropyError)
from learning.architecture import mlp

from learning.testing import helpers


############################
# MLP
############################
def test_MLP_activate_vector():
    model = mlp.MLP((2, 2, 2), transfers=[LinearTransfer(), LinearTransfer()])
    
    # Set weights for deterministic results
    model._bias_vec = numpy.ones(model._bias_vec.shape)
    model._weight_matrices = [numpy.ones(weight_matrix.shape) for weight_matrix in model._weight_matrices]

    # Activate
    assert helpers.approx_equal(model.activate([0, 0]), [2, 2])
    assert helpers.approx_equal(model.activate([0.5, 0.5]), [4, 4])
    assert helpers.approx_equal(model.activate([1, 0]), [4, 4])
    assert helpers.approx_equal(model.activate([0.5, 1]), [5, 5])
    assert helpers.approx_equal(model.activate([1, 1]), [6, 6])


def test_MLP_activate_matrix():
    model = mlp.MLP((2, 2, 2), transfers=[LinearTransfer(), LinearTransfer()])
    
    # Set weights for deterministic results
    model._bias_vec = numpy.ones(model._bias_vec.shape)
    model._weight_matrices = [numpy.ones(weight_matrix.shape) for weight_matrix in model._weight_matrices]

    # Activate
    assert helpers.approx_equal(model.activate([[0, 0], [0.5, 0.5]]), [[2, 2], [4, 4]])
    assert helpers.approx_equal(model.activate([[1, 0], [0.5, 1]]), [[4, 4], [5, 5]])
    assert helpers.approx_equal(model.activate([[1, 1], [0, 0.5]]), [[6, 6], [3, 3]])


# TODO: use validation methods to more robustly test
def test_mlp():
    # Run for a couple of iterations
    # assert that new error is less than original
    model = mlp.MLP((2, 2, 2))
    dataset = datasets.get_xor()

    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=10)
    assert validation.get_error(model, *dataset) < error


@pytest.mark.slowtest
def test_mlp_convergence():
    # Run until convergence
    # assert that network can converge
    model = mlp.MLP((2, 4, 2))
    dataset = datasets.get_xor()

    model.train(*dataset, retries=5, error_break=0.002)
    assert validation.get_error(model, *dataset) <= 0.02


def test_mlp_classifier():
    # Run for a couple of iterations
    # assert that new error is less than original
    model = mlp.MLP(
        (2, 2, 2), transfers=SoftmaxTransfer(), error_func=CrossEntropyError())
    dataset = datasets.get_xor()

    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=20)
    assert validation.get_error(model, *dataset) < error


@pytest.mark.slowtest
def test_mlp_classifier_convergence():
    # Run until convergence
    # assert that network can converge
    model = mlp.MLP(
        (2, 3, 2), transfers=SoftmaxTransfer(), error_func=CrossEntropyError())
    dataset = datasets.get_and()

    model.train(*dataset, retries=5, error_break=0.002)
    assert validation.get_error(model, *dataset) <= 0.02


def test_mlp_perceptron():
    # Given known inputs and weights, test expected outputs
    model = mlp.MLP((2, 1), transfers=mlp.LinearTransfer())
    model._bias_vec[0] = 0.0
    model._weight_matrices[0][0][0] = 0.5
    model._weight_matrices[0][1][0] = -0.5
    assert (model.activate([1, 1]) == [0.0]).all()

    model._weight_matrices[0][0][0] = 1.0
    model._weight_matrices[0][1][0] = 2.0
    assert (model.activate([1, 1]) == [3.0]).all()


def test_mean_list_of_list_of_matrices():
    lol_matrices = [[
        numpy.array([[1, 2], [3, 4]]),
        numpy.array([[-1, -2], [-3, -4]])
    ], [numpy.array([[1, 2], [3, 4]]),
        numpy.array([[1, 2], [3, 4]])]]
    assert helpers.approx_equal(
        mlp._mean_list_of_list_of_matrices(lol_matrices),
        [numpy.array([[1, 2], [3, 4]]),
         numpy.array([[0, 0], [0, 0]])])


def test_mlp_obj_and_obj_jac_match_lin_out_mse():
    _check_obj_and_obj_jac_match(lambda s1, s2, s3: mlp.MLP(
        (s1, s2, s3), transfers=mlp.LinearTransfer(), error_func=MeanSquaredError()))


def test_mlp_obj_and_obj_jac_match_relu_out_ce():
    _check_obj_and_obj_jac_match(
        lambda s1, s2, s3: mlp.MLP(
            (s1, s2, s3), transfers=mlp.ReluTransfer(), error_func=CrossEntropyError()),
        classification=True
    )


def test_mlp_obj_and_obj_jac_match_softmax_out_mse():
    _check_obj_and_obj_jac_match(lambda s1, s2, s3: mlp.MLP(
        (s1, s2, s3), transfers=SoftmaxTransfer(), error_func=MeanSquaredError()))


def test_mlp_obj_and_obj_jac_match_softmax_out_ce():
    _check_obj_and_obj_jac_match(
        lambda s1, s2, s3: mlp.MLP(
            (s1, s2, s3), transfers=SoftmaxTransfer(), error_func=CrossEntropyError()),
        classification=True
    )


def _check_obj_and_obj_jac_match(make_model_func, classification=False):
    """obj and obj_jac functions should return the same obj value."""
    attrs = random.randint(1, 10)
    outs = random.randint(1, 10)
    model = make_model_func(attrs, random.randint(1, 10), outs)

    if classification:
        dataset = datasets.get_random_classification(10, attrs, outs)
    else:
        dataset = datasets.get_random_regression(10, attrs, outs)

    # Don't use exactly the same parameters, to ensure obj functions are actually
    # using the given parameters
    parameters = random.uniform(-1.0, 1.0) * mlp._flatten(
        model._bias_vec, model._weight_matrices)
    assert helpers.approx_equal(
        model._get_obj(parameters, dataset[0], dataset[1]),
        model._get_obj_jac(parameters, dataset[0], dataset[1])[0])


def test_mlp_jacobian_lin_out_mse():
    _check_jacobian(lambda s1, s2, s3: mlp.MLP(
        (s1, s2, s3), transfers=mlp.LinearTransfer(), error_func=MeanSquaredError()))


def test_mlp_jacobian_relu_out_ce():
    _check_jacobian(lambda s1, s2, s3: mlp.MLP(
        (s1, s2, s3), transfers=mlp.ReluTransfer(), error_func=CrossEntropyError()))


def test_mlp_jacobian_softmax_out_mse():
    _check_jacobian(lambda s1, s2, s3: mlp.MLP(
        (s1, s2, s3), transfers=SoftmaxTransfer(), error_func=MeanSquaredError()))


def test_mlp_jacobian_softmax_out_ce():
    _check_jacobian(lambda s1, s2, s3: mlp.MLP(
        (s1, s2, s3), transfers=SoftmaxTransfer(), error_func=CrossEntropyError()))


def _check_jacobian(make_model_func):
    attrs = random.randint(1, 10)
    outs = random.randint(1, 10)

    model = make_model_func(attrs, random.randint(1, 10), outs)
    inp_matrix, tar_matrix = datasets.get_random_regression(random.randint(1, 10), attrs, outs)

    # Test jacobian of error function
    f = lambda xk: model._get_obj(xk, inp_matrix, tar_matrix)
    df = lambda xk: model._get_obj_jac(xk, inp_matrix, tar_matrix)[1]

    helpers.check_gradient(
        f,
        df,
        f_arg_tensor=mlp._flatten(model._bias_vec, model._weight_matrices),
        f_shape='scalar')


def test_MLP_reset():
    shape = (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))

    model = mlp.MLP(shape)
    model_2 = mlp.MLP(shape)

    # Resetting different with the same seed should give the same model
    prev_seed = random.randint(0, 2**32-1)

    try:
        random.seed(0)
        numpy.random.seed(0)
        model.reset()

        random.seed(0)
        numpy.random.seed(0)
        model_2.reset()

        assert model.serialize() == model_2.serialize()
    finally:
        random.seed(prev_seed)
        numpy.random.seed(prev_seed)


##############################
# DropoutMLP
##############################
# TODO: Test dropout MLP jacobians (with inactive neurons)
def test_dropout_mlp():
    # Run for a couple of iterations
    # assert that new error is less than original
    model = mlp.DropoutMLP((2, 8, 2))
    dataset = datasets.get_and()

    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=20)
    assert validation.get_error(model, *dataset) < error


@pytest.mark.slowtest
def test_dropout_mlp_convergence():
    # Run until convergence
    # assert that network can converge
    # Since XOR does not really need dropout, we use high probabilities
    model = mlp.DropoutMLP(
        (2, 8, 2), input_active_probability=1.0, hidden_active_probability=0.9)
    dataset = datasets.get_and()  # Easier and dataset for lienar output

    # Error break lower than cutoff, since dropout may have different error
    # after training
    model.train(*dataset, retries=5, error_break=0.002, error_improve_iters=50)

    # Dropout sacrifices training accuracy for better generalization
    # so we don't worry as much about convergence
    assert validation.get_error(model, *dataset) <= 0.1


def test_dropout_mlp_classifier():
    # Run for a couple of iterations
    # assert that new error is less than original
    model = mlp.DropoutMLP(
        (2, 8, 2), transfers=SoftmaxTransfer(), error_func=CrossEntropyError())
    dataset = datasets.get_and()

    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=20)
    assert validation.get_error(model, *dataset) < error


@pytest.mark.slowtest
def test_dropout_mlp_classifier_convergence():
    # Run until convergence
    # assert that network can converge
    # Since XOR does not really need dropout, we use high probabilities
    model = mlp.DropoutMLP(
        (2, 8, 2),
        transfers=SoftmaxTransfer(),
        error_func=CrossEntropyError(),
        input_active_probability=1.0,
        hidden_active_probability=0.9)
    dataset = datasets.get_and()

    # Error break lower than cutoff, since dropout may have different error
    # after training
    model.train(*dataset, retries=5, error_break=0.002, error_improve_iters=50)

    # Dropout sacrifices training accuracy for better generalization
    # so we don't worry as much about convergence
    assert validation.get_error(model, *dataset) <= 0.1


def test_dropout_mlp_dropout():
    model = mlp.DropoutMLP(
        (2, 4, 3), input_active_probability=0.5, hidden_active_probability=0.5)

    # Only bias and active neurons should not be 0
    model.train_step([[1, 1], [0.5, 0.5]], [[1, 1, 1], [0.5, 0.5, 0.5]])

    # Should still have DropoutTransfers (until activation outside of training)

    _validate_weight_inputs(model._weight_inputs[0],
                            model._input_transfer._active_neurons)
    for weight_inputs, transfer_func in zip(model._weight_inputs[1:-1],
                                            model._transfers[:-1]):
        _validate_weight_inputs(weight_inputs, transfer_func._active_neurons)


def test_dropout_mlp_post_training():
    # Post training should happen on first activate after training (train step),
    # and not more than once, unless training begins again
    model = mlp.DropoutMLP(
        (2, 4, 3), input_active_probability=0.5, hidden_active_probability=0.5)

    # Train, modifying active neurons and weights
    model.train_step([[1, 1], [0.5, 0.5]], [[1, 1, 1], [0.5, 0.5, 0.5]])
    pre_procedure_weights = copy.deepcopy(model._weight_matrices)

    # Should call post_training procedure after activate
    model.activate([1, 1])
    _validate_post_training(model, pre_procedure_weights)

    # Weights should not change after another activate
    model.activate([1, 1])
    _validate_post_training(model, pre_procedure_weights)


def _validate_post_training(model, pre_procedure_weights):
    for weight_matrix, orig_matrix in zip(model._weight_matrices,
                                          pre_procedure_weights):
        # Weights should be scaled by active probability
        assert (weight_matrix == orig_matrix * model._hid_act_prob).all()

    # All inputs and neurons should be active
    assert not isinstance(model._input_transfer, mlp.DropoutTransfer)
    for transfer_func in model._transfers:
        assert not isinstance(transfer_func, mlp.DropoutTransfer)

    for weight_inputs in model._weight_inputs:
        _validate_weight_inputs(weight_inputs, [1.0] * len(weight_inputs))


def _validate_weight_inputs(weight_inputs, active_neurons):
    if len(weight_inputs.shape) == 1:
        weight_inputs = numpy.array([weight_inputs])

    assert weight_inputs.shape[1] == len(active_neurons)

    for input_row in weight_inputs:
        for input_, active in zip(input_row, active_neurons):
            if active == 0.0:
                assert input_ == 0.0
            elif active == 1.0:
                assert input_ != 0.0
            else:
                assert 0, 'Invalid active neuron value'


####################
# DropoutTransfer
####################
def test_dropout_transfer_probability_one():
    length = random.randint(1, 20)

    dropout_transfer = mlp.DropoutTransfer(mlp.LinearTransfer(), 1.0, length)
    assert (dropout_transfer._active_neurons == numpy.array(
        [1.0] * length)).all(), 'All should be active'

    # Random input
    input_vec = numpy.random.random(length)
    assert (dropout_transfer(input_vec) == input_vec).all()


def test_dropout_transfer_probability_zero():
    length = random.randint(1, 20)

    # Can't actually be zero, but can be close enough
    dropout_transfer = mlp.DropoutTransfer(mlp.LinearTransfer(), 1e-16, length)

    # Should not allow zero active, defaults to 1
    assert list(dropout_transfer._active_neurons).count(1.0) == 1
    assert list(dropout_transfer._active_neurons).count(0.0) == length - 1
