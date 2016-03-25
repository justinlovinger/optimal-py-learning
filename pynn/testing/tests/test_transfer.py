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
    layer = transfer.SoftmaxExpTransfer()
    expected = [0.5, 0.5]
    output = list(layer.activate(numpy.array([1.0, 1.0])))
    assert output == expected

    expected = [1.0, 0.0]
    output = list(layer.activate(numpy.array([1.0, 0.0])))
    assert output == expected

    input = [0.75, 0.25]
    output = list(layer.activate(numpy.array(input)))
    assert output[0] > input[0] and output[1] < input[1]

def test_softmax_linear_transfer():
    layer = transfer.SoftmaxLinearTransfer()
    expected = [0.5, 0.5]
    output = list(layer.activate(numpy.array([1.0, 1.0])))
    assert output == expected

    expected = [1.0, 0.0]
    output = list(layer.activate(numpy.array([1.0, 0.0])))
    assert output == expected

    expected = [0.8, 0.2]
    output = list(layer.activate(numpy.array([1.0, 0.25])))
    assert output == expected