import numpy

from pynn.architecture import transfer
from pynn.testing import helpers

#####################
# Tanh
#####################
def test_tanh_transfer():
    layer = transfer.TanhTransfer()
    expected = [-0.761594, 0.0, 0.462117, 0.761594]
    output = layer.activate(numpy.array([-1.0, 0.0, 0.5, 1.0]))
    output = [round(v, 6) for v in output]
    assert output == expected


#####################
# Gaussian
#####################
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


#####################
# Softmax
#####################
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


###############
# Normalize
###############
def test_normalize_transfer():
    layer = transfer.NormalizeTransfer()

    # Passing inputs as scaling inputs allows normalization to 1.0
    inputs = numpy.array([1.0, 1.0])
    assert list(layer.activate(inputs, inputs)) == [0.5, 0.5]

    inputs = numpy.array([1.0, 0.0])
    assert list(layer.activate(inputs, inputs)) == [1.0, 0.0]

    inputs = numpy.array([1.0, 0.25])
    assert list(layer.activate(inputs, inputs)) == [0.8, 0.2]

    # Non same scaling inputs
    assert list(layer.activate(numpy.array([1.0, 0.5]), numpy.array([0.75, 0.25]))) == [1.0, 0.5]
    assert list(layer.activate(numpy.array([1.0, 0.5]), numpy.array([1.75, 0.25]))) == [0.5, 0.25]


##############
# ReLU
##############
def test_relu_transfer():
    layer = transfer.ReluTransfer()

    assert helpers.approx_equal(list(layer.activate(numpy.array([0, 1]))),
                                [0.6931471805, 1.3132616875])
    assert helpers.approx_equal(list(layer.activate(numpy.array([-1.5, 10]))),
                                [0.201413, 10.00004539])


def test_relu_derivative():
    assert helpers.approx_equal(list(transfer.drelu(numpy.array([0, 1]))),
                                [0.5, 0.73105857])
    assert helpers.approx_equal(list(transfer.drelu(numpy.array([-1.5, 10]))),
                                [0.182426, 0.9999546])