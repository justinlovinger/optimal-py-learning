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
