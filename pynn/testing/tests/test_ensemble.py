from pynn.architecture import ensemble
from pynn.testing import helpers

def test_bagger():
    # Create dummy layers that return set outputs
    outputs = [[0, 1, 2], [1, 2, 3]]
    models = [helpers.SetOutputModel(output) for output in outputs]
    bagger = ensemble.Bagger(models)

    # Assert bagger returns average of those outputs
    output = bagger.activate([])
    assert list(output) == [0.5, 1.5, 2.5]