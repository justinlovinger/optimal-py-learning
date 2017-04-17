import pytest
import copy
import random

import numpy

from pynn import graph
from pynn import network
from pynn.data import datasets
from pynn.architecture import mlp, rbf
from pynn.testing import helpers

def test_mse():
    # This network will always output 0 for input 0
    nn = helpers.SetOutputModel(0)
    assert nn.mse([[0], [1]]) == 1.0
    assert nn.mse([[0], [0.5]]) == 0.25

    nn = helpers.SetOutputModel([0, 0])
    assert nn.mse([[0], [1, 1]]) == 1.0


def test_post_pattern_callback():
    pat = datasets.get_xor()
    nn = helpers.EmptyModel()

    history = []
    def callback(nn, pattern):
        history.append(pattern)

    nn.train(pat, iterations=1, post_pattern_callback=callback)
    assert pat == history

##########################
# Full architecture tests
##########################
# TODO: use validation methods to more robustly test
def test_mlp():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = mlp.MLP((2, 2, 1))
    pat = datasets.get_xor()

    error = nn.avg_mse(pat)
    nn.train(pat, 10)
    assert nn.avg_mse(pat) < error


pytest.mark.slowtest()
def test_mlp_convergence():
    # Run until convergence
    # assert that network can converge
    nn = mlp.MLP((2, 2, 2, 1))
    pat = datasets.get_xor()

    nn.train(pat, error_break=0.015)
    assert nn.avg_mse(pat) <= 0.02


def test_mlp_classifier():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = mlp.MLP((2, 2, 2), transfers=mlp.SoftmaxTransferPerceptron())
    pat = datasets.get_xor()
    _make_xor_one_hot(pat)

    error = nn.avg_mse(pat)
    nn.train(pat, 10)
    assert nn.avg_mse(pat) < error


pytest.mark.slowtest()
def test_mlp_classifier_convergence():
    # Run until convergence
    # assert that network can converge
    nn = mlp.MLP((2, 2, 2), transfers=mlp.SoftmaxTransferPerceptron(),
                 learn_rate=0.01, momentum_rate=0.005)
    pat = datasets.get_and()
    _make_xor_one_hot(pat)

    nn.train(pat, error_break=0.015)
    assert nn.avg_mse(pat) <= 0.02


def test_dropout_mlp():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = mlp.DropoutMLP((2, 2, 1))
    pat = datasets.get_xor()

    error = nn.avg_mse(pat)
    nn.train(pat, 10)
    assert nn.avg_mse(pat) < error


pytest.mark.slowtest()
def test_dropout_mlp_convergence():
    # Run until convergence
    # assert that network can converge
    # Since XOR does not really need dropout, we use high probabilities
    nn = mlp.DropoutMLP((2, 6, 3, 1), learn_rate=0.2, momentum_rate=0.1,
                        input_active_probability=1.0,
                        hidden_active_probability=0.9)
    pat = datasets.get_xor()

    # Error break lower than cutoff, since dropout may have different error
    # after training
    nn.train(pat, error_break=0.002, pattern_select_func=network.select_sample)

    # Dropout sacrifices training accuracy for better generalization
    # so we don't worry as much about convergence
    assert nn.avg_mse(pat) <= 0.1


def test_dropout_mlp_classifier():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = mlp.DropoutMLP((2, 6, 3, 2), transfers=mlp.SoftmaxTransferPerceptron(),
                        learn_rate=0.2, momentum_rate=0.1)
    pat = datasets.get_and()
    _make_xor_one_hot(pat)

    error = nn.avg_mse(pat)
    nn.train(pat, 10, pattern_select_func=network.select_sample)
    assert nn.avg_mse(pat) < error


pytest.mark.slowtest()
def test_dropout_mlp_classifier_convergence():
    # Run until convergence
    # assert that network can converge
    # Since XOR does not really need dropout, we use high probabilities
    nn = mlp.DropoutMLP((2, 6, 3, 2), transfers=mlp.SoftmaxTransferPerceptron(),
                        learn_rate=0.2, momentum_rate=0.1,
                        input_active_probability=1.0,
                        hidden_active_probability=0.9)
    pat = datasets.get_and()
    _make_xor_one_hot(pat)

    # Error break lower than cutoff, since dropout may have different error
    # after training
    nn.train(pat, error_break=0.002)

    # Dropout sacrifices training accuracy for better generalization
    # so we don't worry as much about convergence
    assert nn.avg_mse(pat) <= 0.1


def _make_xor_one_hot(dataset):
    # TODO: make a function in process.py to automatically do this
    for pattern in dataset:
        if pattern[1][0] == 0.0:
            pattern[1] = [1, 0]
        elif pattern[1][0] == 1.0:
            pattern[1] = [0, 1]
        else:
            raise ValueError()

def test_rbf():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = rbf.RBF(2, 4, 1, scale_by_similarity=True)
    pat = datasets.get_xor()

    error = nn.avg_mse(pat)
    nn.train(pat, 10)
    assert nn.avg_mse(pat) < error


pytest.mark.slowtest()
def test_rbf_convergence():
    # Run until convergence
    # assert that network can converge
    nn = rbf.RBF(2, 4, 1, scale_by_similarity=True)
    pat = datasets.get_xor()

    nn.train(pat, error_break=0.015)
    assert nn.avg_mse(pat) <= 0.02


################################
# Datapoint selection functions
################################
@pytest.fixture()
def seed_random(request):
    random.seed(0)

    def fin():
        import time
        random.seed(time.time())
    request.addfinalizer(fin)


def test_select_sample(seed_random):
    pat = datasets.get_xor()
    new_pat = network.select_sample(pat)
    assert len(new_pat) == len(pat)
    for p in pat: # all in
        assert p in new_pat
    assert new_pat != pat # order different

    new_pat = network.select_random(pat, size=2)
    assert len(new_pat) == 2
    # No duplicates
    count = 0
    for p in pat:
        if p in new_pat:
            count += 1
    assert count == 2


def test_select_random(monkeypatch):
    # Monkeypatch so we know that random returns
    monkeypatch.setattr(random, 'randint', lambda x, y : 0) # randint always returns 0

    pat = datasets.get_xor()
    new_pat = network.select_random(pat)
    assert len(new_pat) == len(pat)
    for p in new_pat:
        assert p == pat[0] # due to monkeypatch

    new_pat = network.select_random(pat, size=2)
    assert len(new_pat) == 2
    for p in new_pat:
        assert p == pat[0]

#########################
# Pre and post hooks
#########################
class CountMLP(mlp.MLP):
    def __init__(self, *args, **kwargs):
        super(CountMLP, self).__init__(*args, **kwargs)
        self.count = 0


def test_pre_iteration():
    # Setup pre_iteration function
    class TestMLP(CountMLP):
        def pre_iteration(self, patterns):
            self.count += 1

    # Train for a few iterations
    nn = TestMLP((1, 1))
    nn.train([[[1], [1]]], iterations=10, error_break=None)

    # Count incremented for each iteration
    assert nn.count == 10


def test_post_iteration():
    # Setup post_iteration function
    class TestMLP(CountMLP):
        def post_iteration(self, patterns):
            self.count += 1

    # Train for a few iterations
    nn = TestMLP((1, 1))
    nn.train([[[1], [1]]], iterations=10, error_break=None)

    # Count incremented for each iteration
    assert nn.count == 10

####################
# Train function
####################
def test_break_on_stagnation_completely_stagnant():
    # If error doesn't change by enough after enough iterations
    # stop training

    nn = helpers.SetOutputModel(1.0)

    # Stop training if error does not change by more than threshold after
    # distance iterations
    nn.train([([0.0], [0.0])], error_stagnant_distance=5, error_stagnant_threshold=0.01)
    assert nn.iteration == 6 # The 6th is 5 away from the first

def test_break_on_stagnation_dont_break_if_wrapped_around():
    # Should not break on situations like: 1.0, 0.9, 0.8, 0.7, 1.0
    # Since error did change, even if it happens to be the same after
    # n iterations
    nn = helpers.ManySetOutputsModel([[1.0], [0.9], [0.8], [0.7], [1.0], [1.0], [1.0], [1.0], [1.0]])

    # Should pass wrap around to [1.0], and stop after consecutive [1.0]s
    nn.train([([0.0], [0.0])], error_stagnant_distance=4, error_stagnant_threshold=0.01)
    assert nn.iteration == 9