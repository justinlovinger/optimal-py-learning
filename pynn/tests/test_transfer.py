import numpy

from pynn import transfer

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

def test_softmax_exp_transfer():
    assert 0

def test_softmax_linear_transfer():
    assert 0