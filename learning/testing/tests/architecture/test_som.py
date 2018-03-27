###############################################################################
# The MIT License (MIT)
#
# Copyright (c) 2017 Justin Lovinger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import random
import math

import numpy

from learning import datasets
from learning.architecture import som

from learning.testing import helpers

def test_SOM_activate_vector():
    som_ = som.SOM(2, 2)
    som_._weights = numpy.ones(som_._weights.shape)

    assert helpers.approx_equal(som_.activate([1, 1]), [0, 0])
    assert helpers.approx_equal(som_.activate([0, 1]), [1, 1])
    assert helpers.approx_equal(som_.activate([1, 0]), [1, 1])
    assert helpers.approx_equal(som_.activate([1, 2]), [1, 1])
    assert helpers.approx_equal(som_.activate([1, 3]), [2, 2])
    assert helpers.approx_equal(som_.activate([0.2, 1.6]), [1, 1])
    assert helpers.approx_equal(som_.activate([1.8, 1.6]), [1, 1])


def test_SOM_activate_matrix():
    som_ = som.SOM(2, 2)
    som_._weights = numpy.ones(som_._weights.shape)

    assert helpers.approx_equal(som_.activate([[1, 1], [0, 1]]), [[0, 0], [1, 1]])
    assert helpers.approx_equal(som_.activate([[1, 0], [1, 2]]), [[1, 1], [1, 1]])
    assert helpers.approx_equal(som_.activate([[1, 3], [0.2, 1.6]]), [[2, 2], [1, 1]])
    assert helpers.approx_equal(som_.activate([[1.8, 1.6], [0, 0]]), [[1, 1], [1.4142135623730951, 1.4142135623730951]])


def test_som_reduces_distances_vector():
    # SOM functions correctly if is moves neurons towards inputs
    input_matrix, target_matrix = datasets.get_xor()

    # Small initial weight range chosen so network isn't "accidentally"
    # very close to inputs initially (which could cause test to fail)
    num_neurons = random.randint(2, 10)
    som_ = som.SOM(2, num_neurons, initial_weights_range=0.25)

    # Convenience function
    def min_distances():
        all_closest = []
        for inp_vec in input_matrix:
            distances = som_.activate(inp_vec)
            all_closest.append(min(distances))
        return all_closest

    # Train SOM
    # Assert that distances have decreased
    all_closest = min_distances()
    som_.train(input_matrix, target_matrix, iterations=20)
    new_closest = min_distances()
    print all_closest
    print new_closest
    for old_c, new_c in zip(all_closest, new_closest):
        assert new_c < old_c


def test_som_reduces_distances_matrix():
    # SOM functions correctly if is moves neurons towards inputs
    input_matrix, target_matrix = datasets.get_xor()

    # Small initial weight range chosen so network isn't "accidentally"
    # very close to inputs initially (which could cause test to fail)
    num_neurons = random.randint(2, 10)
    som_ = som.SOM(2, num_neurons, initial_weights_range=0.25)

    # Convenience function
    def min_distances():
        distance_matrix = som_.activate(input_matrix)
        assert distance_matrix.shape == (len(input_matrix[0]), num_neurons)
        return numpy.min(distance_matrix, axis=-1)

    # Train SOM
    # Assert that distances have decreased
    all_closest = min_distances()
    som_.train(input_matrix, target_matrix, iterations=20)
    new_closest = min_distances()
    print all_closest
    print new_closest
    for old_c, new_c in zip(all_closest, new_closest):
        assert new_c < old_c
