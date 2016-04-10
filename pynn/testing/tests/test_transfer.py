import numpy

from pynn import transfer
from pynn.testing import helpers

def test_sigmoid_transfer():
    layer = transfer.SigmoidTransfer()
    expected = [-0.761594, 0.0, 0.462117, 0.761594]
    output = layer.activate(numpy.array([-1.0, 0.0, 0.5, 1.0]))
    output = [round(v, 6) for v in output]
    assert output == expected

def test_gaussian_transfer():
    layer = transfer.GaussianTransfer()
    expected = [0.367879, 1.0, 0.778801, 0.367879]
    output = layer.activate(numpy.array([-1.0, 0.0, 0.5, 1.0]))
    output = [round(v, 6) for v in output]
    assert output == expected

    layer = transfer.GaussianTransfer(variance=0.5)
    expected = [0.135335, 1.0, 0.606531, 0.135335]
    output = layer.activate(numpy.array([-1.0, 0.0, 0.5, 1.0]))
    output = [round(v, 6) for v in output]
    assert output == expected

def test_softmax_transfer():
    layer = transfer.SoftmaxTransfer()

    assert list(layer.activate(numpy.array([1.0, 1.0]))) == [0.5, 0.5]

    expecteds = [0.7310585, 0.2689414]
    outputs = list(layer.activate(numpy.array([1.0, 0.0])))
    for output, expected in zip(outputs, expecteds):
        assert helpers.approx_equal(output, expected)

    output_ = list(layer.activate(numpy.array([1.0, -1.0])))
    assert output_[0] > 0.5 and output_[1] < 0.5
    assert sum(output_) == 1.0