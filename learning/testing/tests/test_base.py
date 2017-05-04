import pytest
import copy
import random

import numpy

from learning import graph
from learning import base
from learning.data import datasets
from learning.architecture import mlp, rbf

from learning.testing import helpers

def test_mse():
    # This network will always output 0 for input 0
    nn = helpers.SetOutputModel(0)
    assert nn.mse([0], [1]) == 1.0
    assert nn.mse([0], [0.5]) == 0.25

    nn = helpers.SetOutputModel([0, 0])
    assert nn.mse([0], [1, 1]) == 1.0


def test_post_pattern_callback():
    dataset = datasets.get_xor()
    model = helpers.EmptyModel()

    inp_history = []
    tar_history = []
    def callback(model, input_vec, target_vec):
        inp_history.append(input_vec)
        tar_history.append(target_vec)

    model.train(*dataset, iterations=1, post_pattern_callback=callback)
    inp_history = numpy.array(inp_history)
    tar_history = numpy.array(tar_history)
    assert (dataset[0] == inp_history).all()
    assert (dataset[1] == tar_history).all()


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


def test_select_sample_size_none(seed_random):
    input_matrix, target_matrix = datasets.get_xor()

    new_inp_matrix, new_tar_matrix = base.select_sample(input_matrix, target_matrix)
    assert new_inp_matrix.shape == input_matrix.shape
    assert new_tar_matrix.shape == target_matrix.shape

    for inp_vec in input_matrix: # all in
        assert inp_vec in new_inp_matrix
    for tar_vec in target_matrix: # all in
        assert tar_vec in new_tar_matrix

    assert not (new_inp_matrix == input_matrix).all() # order different
    assert not (new_tar_matrix == target_matrix).all() # order different

def test_select_sample(seed_random):
    input_matrix, target_matrix = datasets.get_xor()

    # Test size param
    new_inp_matrix, new_tar_matrix = base.select_sample(input_matrix, target_matrix, size=2)
    assert new_inp_matrix.shape[0] == 2
    assert new_tar_matrix.shape[0] == 2

    # No duplicates
    count = 0
    for inp_vec in new_inp_matrix:
        if inp_vec in input_matrix:
            count += 1
    assert count == 2

    count = 0
    for tar_vec in new_tar_matrix:
        if tar_vec in target_matrix:
            count += 1
    assert count == 2


def test_select_random_size_none(monkeypatch):
    # Monkeypatch so we know that random returns
    monkeypatch.setattr(random, 'randint', lambda x, y : 0) # randint always returns 0

    input_matrix, target_matrix = datasets.get_xor()
    new_inp_matrix, new_tar_matrix = base.select_random(input_matrix, target_matrix)
    assert new_inp_matrix.shape == input_matrix.shape
    assert new_tar_matrix.shape == target_matrix.shape

    for inp_vec in new_inp_matrix:
        assert (inp_vec == input_matrix[0]).all() # due to monkeypatch
    for tar_vec in new_tar_matrix:
        assert (tar_vec == target_matrix[0]).all() # due to monkeypatch

def test_select_random(monkeypatch):
    # Monkeypatch so we know that random returns
    monkeypatch.setattr(random, 'randint', lambda x, y : 0) # randint always returns 0

    input_matrix, target_matrix = datasets.get_xor()

    # Test size param
    new_inp_matrix, new_tar_matrix = base.select_random(input_matrix, target_matrix, size=2)

    assert new_inp_matrix.shape[0] == 2
    assert new_tar_matrix.shape[0] == 2

    for inp_vec in new_inp_matrix:
        assert (inp_vec == input_matrix[0]).all() # due to monkeypatch
    for tar_vec in new_tar_matrix:
        assert (tar_vec == target_matrix[0]).all() # due to monkeypatch

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
        def pre_iteration(self, input_matrix, target_matrix):
            self.count += 1

    # Train for a few iterations
    nn = TestMLP((1, 1))
    nn.train([1], [1], iterations=10, error_break=None)

    # Count incremented for each iteration
    assert nn.count == 10


def test_post_iteration():
    # Setup post_iteration function
    class TestMLP(CountMLP):
        def post_iteration(self, input_matrix, target_matrix):
            self.count += 1

    # Train for a few iterations
    nn = TestMLP((1, 1))
    nn.train([[1]], [[1]], iterations=10, error_break=None)

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
    nn.train([[0.0]], [[0.0]], error_stagnant_distance=5, error_stagnant_threshold=0.01)
    assert nn.iteration == 6 # The 6th is 5 away from the first

def test_break_on_stagnation_dont_break_if_wrapped_around():
    # Should not break on situations like: 1.0, 0.9, 0.8, 0.7, 1.0
    # Since error did change, even if it happens to be the same after
    # n iterations
    nn = helpers.ManySetOutputsModel([[1.0], [0.9], [0.8], [0.7], [1.0], [1.0], [1.0], [1.0], [1.0]])

    # Should pass wrap around to [1.0], and stop after consecutive [1.0]s
    nn.train([[0.0]], [[0.0]], error_stagnant_distance=4, error_stagnant_threshold=0.01)
    assert nn.iteration == 9

@pytest.mark.skip(reason='Hard to test, but not hard to implement')
def test_model_train_retry():
    # Model should reset and retry if it doesn't converge
    # Train should not calculate avg_mse if it is out of retries
    assert 0
