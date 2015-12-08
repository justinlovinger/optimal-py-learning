import pytest

from pynn import network

def test_network_validation_layers():
    with pytest.raises(ValueError):
        n = network.Network([None])

    with pytest.raises(ValueError):
        n = network.Network([1])

    with pytest.raises(ValueError):
        n = network.Network(['q'])

    n = network.Network([network.Layer()])

def test_network_validation_required_next_prev():
    with pytest.raises(ValueError):
        n = network.Network([network.GrowingLayer(), network.Layer()])

    n = network.Network([network.GrowingLayer(), network.SupportsGrowingLayer()])