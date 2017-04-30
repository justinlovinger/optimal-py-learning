import numpy

from pynn.architecture import knn

from pynn.testing import helpers

def test_select_k_nearest_neighbors():
    points = [((0,), (0,)), ((1,), (0,)), ((2,), (0,))]
    center = [0]

    assert set(knn.select_k_nearest_neighbors(points, center, 2)) == set([((0,), (0,)), ((1,), (0,))])
    assert set(knn.select_k_nearest_neighbors(points, center, 1)) == set([((0,), (0,))])

    points = [((0, 0), (0,)), ((1, 1), (0,)), ((2, 2), (0,))]
    center = [0, 0]

    assert set(knn.select_k_nearest_neighbors(points, center, 2)) == set([((0, 0), (0,)), ((1, 1), (0,))])
    assert set(knn.select_k_nearest_neighbors(points, center, 3)) == set([((0, 0), (0,)), ((1, 1), (0,)), ((2, 2), (0,))])

def test_select_k_nearest_neighbors_numpy_array_inputs():
    points = [(numpy.array([0]), (0,)),
              (numpy.array([1]), (0,)),
              (numpy.array([2]), (0,))]
    center = numpy.array([0])

    assert helpers.equal_ignore_order(knn.select_k_nearest_neighbors(points, center, 2),
                                      [(numpy.array([0]), (0,)),
                                       (numpy.array([1]), (0,))])
    assert helpers.equal_ignore_order(knn.select_k_nearest_neighbors(points, center, 1), 
                                      [(numpy.array([0]), (0,))])

    points = [(numpy.array([0, 0]), (0,)),
              (numpy.array([1, 1]), (0,)),
              (numpy.array([2, 2]), (0,))]
    center = numpy.array([0, 0])

    assert helpers.equal_ignore_order(knn.select_k_nearest_neighbors(points, center, 2),
                                      [(numpy.array([0, 0]), (0,)),
                                       (numpy.array([1, 1]), (0,)),])
    assert helpers.equal_ignore_order(knn.select_k_nearest_neighbors(points, center, 3),
                                      [(numpy.array([0, 0]), (0,)),
                                       (numpy.array([1, 1]), (0,)),
                                       (numpy.array([2, 2]), (0,))])