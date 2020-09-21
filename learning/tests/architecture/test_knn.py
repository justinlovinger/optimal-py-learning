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

import numpy

from learning.architecture import knn

from learning.testing import helpers


def test_select_k_nearest_neighbors():
    matrix = [(0,), (1,), (2,)]
    center = [0]

    assert set(knn.select_k_nearest_neighbors(matrix, center, 2)) == set([0, 1])
    assert set(knn.select_k_nearest_neighbors(matrix, center, 1)) == set([0])

    matrix = [(0, 0), (1, 1), (2, 2)]
    center = [0, 0]

    assert set(knn.select_k_nearest_neighbors(matrix, center, 2)) == set([0, 1])
    assert set(knn.select_k_nearest_neighbors(matrix, center, 3)) == set([0, 1, 2])


def test_select_k_nearest_neighbors_numpy_matrix():
    matrix = numpy.array([(0,), (1,), (2,)])
    center = numpy.array([0])

    assert set(knn.select_k_nearest_neighbors(matrix, center, 2)) == set([0, 1])
    assert set(knn.select_k_nearest_neighbors(matrix, center, 1)) == set([0])

    matrix = numpy.array([(0, 0), (1, 1), (2, 2)])
    center = numpy.array([0, 0])

    assert set(knn.select_k_nearest_neighbors(matrix, center, 2)) == set([0, 1])
    assert set(knn.select_k_nearest_neighbors(matrix, center, 3)) == set([0, 1, 2])
