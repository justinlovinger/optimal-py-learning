import pytest
import copy
import random

from pynn import graph
from pynn import network
from pynn.data import datasets
from pynn.architecture import mlp


def test_layer_as_key():
    layer = network.Layer()
    layer2 = network.Layer()
    dict_ = {layer: layer2,
             layer2: layer}

    assert dict_[layer] == layer2
    assert dict_[layer2] == layer


def test_incoming_order_dict():
    incoming_layers = []
    for i in range(20):
        incoming_layers.append(network.Layer())

    # Construct network with many layers going into one
    last_layer = network.Layer()
    layers = {'I': incoming_layers,
              last_layer: ['O']}
    for layer in incoming_layers:
        layers[layer] = [last_layer]

    incoming_order = {last_layer: incoming_layers} # Same order as we constructed
    nn = network.Network(layers, incoming_order_dict=incoming_order)

    assert nn._graph.backwards_adjacency[last_layer] == incoming_layers


def test_activation_order():
    adjacency_dict = {'I': ['a'],
                      '1': ['a'],
                      'a': ['b', 'c'],
                      'b': ['c'],
                      'c': ['O']}
    activation_order = network._make_activation_order(graph.Graph(adjacency_dict))
    assert activation_order == ['1', 'a', 'b', 'c']


def test_activation_order_cycle():
    adjacency_dict = {'I': ['a'],
                      'a': ['b'],
                      'b': ['1', 'O'],
                      '1': ['a']}
    activation_order = network._make_activation_order(graph.Graph(adjacency_dict))
    assert activation_order == ['a', 'b', '1']



def test_network_validation_layers():
    with pytest.raises(TypeError):
        n = network.Network([None])

    with pytest.raises(TypeError):
        n = network.Network([1])

    with pytest.raises(TypeError):
        n = network.Network(['q'])

    n = network.Network([network.Layer()])


def test_network_validation_requires_next_prev():
    with pytest.raises(TypeError):
        n = network.Network([network.GrowingLayer(), network.Layer()])

    n = network.Network([network.GrowingLayer(), network.SupportsGrowingLayer()])


def test_network_validation_parallel_requires_prev_next():
    layer = network.Layer()
    layer.attributes = ['test', 'test2']

    layers = []
    for i in range(3):
        layers.append(copy.deepcopy(layer))

    prev_layer = network.Layer()
    prev_layer.requires_next = ['test']
    next_layer = network.Layer()
    next_layer.requires_prev = ['test2']

    # All layers same: valid
    network._validate_layers_parallel(layers, prev_layer, next_layer)

    # Change 1: invalid
    layers[0].attributes = []
    with pytest.raises(TypeError):
        network._validate_layers_parallel(layers, prev_layer, next_layer)


def test_get_error():
    # This network will always output 0 for input 0
    nn = network.Network([mlp.Perceptron(1, 1)])
    assert nn.get_error([[0], [1]]) == 1.0
    assert nn.get_error([[0], [0.5]]) == 0.25

    nn = network.Network([mlp.Perceptron(1, 2)])
    assert nn.get_error([[0], [1, 1]]) == 1.0


def test_post_pattern_callback():
    pat = datasets.get_xor()
    nn = network.Network([])

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
    nn = network.make_mlp((2, 2, 1))
    pat = datasets.get_xor()

    error = nn.get_avg_error(pat)
    nn.train(pat, 10)
    assert nn.get_avg_error(pat) < error


pytest.mark.slowtest()
def test_mlp_convergence():
    # Run until convergence
    # assert that network can converge
    nn = network.make_mlp((2, 2, 1))
    pat = datasets.get_xor()

    cutoff = 0.02
    nn.train(pat, error_break=0.02)
    assert nn.get_avg_error(pat) <= cutoff


def test_rbf():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = network.make_rbf(2, 4, 1, normalize=True)
    pat = datasets.get_xor()

    error = nn.get_avg_error(pat)
    nn.train(pat, 10)
    assert nn.get_avg_error(pat) < error


pytest.mark.slowtest()
def test_rbf_convergence():
    # Run until convergence
    # assert that network can converge
    nn = network.make_rbf(2, 4, 1, normalize=True)
    pat = datasets.get_xor()

    cutoff = 0.02
    nn.train(pat, error_break=0.02)
    assert nn.get_avg_error(pat) <= cutoff


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
class CountPerceptron(mlp.Perceptron):
    def __init__(self, *args, **kwargs):
        super(CountPerceptron, self).__init__(*args, **kwargs)
        self.count = 0


def test_pre_training():
    # Setup pre_training function
    class TestPerceptron(CountPerceptron):
        def pre_training(self, patterns):
            self.count += 1

    # Train for a few iterations
    nn = network.Network([TestPerceptron(1, 1)])
    nn.train([[[1], [1]]], iterations=10, error_break=None)

    # Count incremented only once
    assert list(nn._activation_order)[0].count == 1


def test_post_training():
    # Setup post_training function
    class TestPerceptron(CountPerceptron):
        def post_training(self, patterns):
            self.count += 1

    # Train for a few iterations
    nn = network.Network([TestPerceptron(1, 1)])
    nn.train([[[1], [1]]], iterations=10, error_break=None)

    # Count incremented only once
    assert list(nn._activation_order)[0].count == 1


def test_pre_iteration():
    # Setup pre_iteration function
    class TestPerceptron(CountPerceptron):
        def pre_iteration(self, patterns):
            self.count += 1

    # Train for a few iterations
    nn = network.Network([TestPerceptron(1, 1)])
    nn.train([[[1], [1]]], iterations=10, error_break=None)

    # Count incremented for each iteration
    assert list(nn._activation_order)[0].count == 10


def test_post_iteration():
    # Setup post_iteration function
    class TestPerceptron(CountPerceptron):
        def post_iteration(self, patterns):
            self.count += 1

    # Train for a few iterations
    nn = network.Network([TestPerceptron(1, 1)])
    nn.train([[[1], [1]]], iterations=10, error_break=None)

    # Count incremented for each iteration
    assert list(nn._activation_order)[0].count == 10