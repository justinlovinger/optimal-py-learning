from pynn import ensemble
from pynn import network
from pynn.testing import helpers

def test_bagger():
    # Create dummy layers that return set outputs
    outputs = [[0, 1, 2], [1, 2, 3]]
    layers = [helpers.SetOutputLayer(output) for output in outputs]
    networks = [network.Network([layer]) for layer in layers]
    bagger = ensemble.Bagger(networks)

    # Assert bagger returns average of those outputs
    output = bagger.activate([])
    assert list(output) == [0.5, 1.5, 2.5]