import numpy

from pynn import transform

def test_perceptron():
    # Given known inputs, test expected outputs
    layer = transform.Perceptron(2, 1)
    layer._weights[0][0] = 0.5
    layer._weights[1][0] = -0.5
    assert layer.activate(numpy.array([1, 1])) == 0.0

    layer._weights[0][0] = 1.0
    layer._weights[1][0] = 2.0
    assert layer.activate(numpy.array([1, 1])) == 3.0

def test_add_bias():
    # Given 0 vector inputs, output should not be 0
    bias = transform.AddBias(transform.Perceptron(2, 1))
    assert bias.activate(numpy.array([0]))[0] != 0.0

    # Without bias, output should be 0
    layer = transform.Perceptron(1, 1)
    assert layer.activate(numpy.array([0]))[0] == 0.0

def test_gaussian_output():
    # Given known inputs, test expected outputs
    layer = transform.GaussianOutput(2, 1)
    layer._weights[0][0] = 0.5
    layer._weights[1][0] = -0.5
    assert layer.activate(numpy.array([1, 1])) == 0.0

    layer._weights[0][0] = 1.0
    layer._weights[1][0] = 2.0
    assert layer.activate(numpy.array([1, 1])) == 3.0