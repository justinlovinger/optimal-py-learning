from pynn.architecture import knn

def test_select_k_nearest_neighbors():
    points = [((0,), (0,)), ((1,), (0,)), ((2,), (0,))]
    center = [0]

    assert set(knn.select_k_nearest_neighbors(points, center, 2)) == set([((0,), (0,)), ((1,), (0,))])
    assert set(knn.select_k_nearest_neighbors(points, center, 1)) == set([((0,), (0,))])

    points = [((0, 0), (0,)), ((1, 1), (0,)), ((2, 2), (0,))]
    center = [0, 0]

    assert set(knn.select_k_nearest_neighbors(points, center, 2)) == set([((0, 0), (0,)), ((1, 1), (0,))])
    assert set(knn.select_k_nearest_neighbors(points, center, 3)) == set([((0, 0), (0,)), ((1, 1), (0,)), ((2, 2), (0,))])