import numpy

from pynn import network
from pynn import ensemble
from pynn.testing import helpers

def test_bagger():
    # Create bagger with dummy layers that return set outputs
    outputs = [[0, 1, 2], [1, 2, 3]]
    layers = [helpers.SetOutputLayer(o) for o in outputs]
    bagger = ensemble.Bagger(layers)
    nn = network.Network([bagger])

    # Assert bagger returns average of those outputs
    output = nn.activate([])
    print output
    assert list(output) == [0.5, 1.5, 2.5]