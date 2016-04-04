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

def test_select_k_nearest_neighbors():
    inputs = [(0,), (1,), (2,)]
    center = [0]

    assert set(transform.select_k_nearest_neighbors(inputs, center, 2)) == set([(0,), (1,)])
    assert set(transform.select_k_nearest_neighbors(inputs, center, 1)) == set([(0,)])

    inputs = [(0, 0), (1, 1), (2, 2)]
    center = [0, 0]

    assert set(transform.select_k_nearest_neighbors(inputs, center, 2)) == set([(0, 0), (1, 1)])
    assert set(transform.select_k_nearest_neighbors(inputs, center, 3)) == set([(0, 0), (1, 1), (2, 2)])