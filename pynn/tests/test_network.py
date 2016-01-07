import pytest
import copy

from pynn import network
from pynn.data import datasets

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

def test_network_validation_sequence_num_inputs_outputs():
    layer = network.Layer()
    layer.num_inputs = 3
    layer.num_outputs = 2

    layer2 = network.Layer()
    layer2.num_inputs = 2
    layer2.num_outputs = 1
    
    layer3 = network.Layer()
    layer3.num_inputs = 1
    layer3.num_outputs = 1

    layers = [layer, layer2, layer3]

    # Layers line up: valid
    network._validate_layers_sequence(layers)

    # Change 1: invalid
    layers[0].num_outputs = 1
    with pytest.raises(ValueError):
        network._validate_layers_sequence(layers)

def test_network_validation_parallel_num_inputs_outputs():
    layer = network.Layer()
    layer.num_inputs = 2
    layer.num_outputs = 1

    layers = []
    for i in range(3):
        layers.append(copy.deepcopy(layer))

    # All layers same: valid
    network._validate_layers_parallel(layers, None, None)

    # Change num inputs for 1: invalid
    layers[0].num_inputs = 1
    with pytest.raises(ValueError):
        network._validate_layers_parallel(layers, None, None)

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

def test_mlp():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = network.make_mlp((2, 2, 1))
    pat = datasets.get_xor()

    error = nn.get_error(pat)
    nn.train(pat, 10)
    assert nn.get_error(pat) < error

pytest.mark.slowtest()
def test_mlp_convergence():
    # Run until convergence
    # assert that network can converge
    nn = network.make_mlp((2, 2, 1))
    pat = datasets.get_xor()

    cutoff = 0.02
    nn.train(pat, error_break=0.02)
    assert nn.get_error(pat) <= cutoff

def test_rbf():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = network.make_rbf(2, 4, 1, normalize=True)
    pat = datasets.get_xor()

    error = nn.get_error(pat)
    nn.train(pat, 10)
    assert nn.get_error(pat) < error

pytest.mark.slowtest()
def test_rbf_convergence():
    # Run until convergence
    # assert that network can converge
    nn = network.make_rbf(2, 4, 1, normalize=True)
    pat = datasets.get_xor()

    cutoff = 0.02
    nn.train(pat, error_break=0.02)
    assert nn.get_error(pat) <= cutoff