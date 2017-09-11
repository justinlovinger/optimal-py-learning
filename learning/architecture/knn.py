"""Layers and functions for a k-nearest-neighbors architecture."""
import heapq

from learning import calculate

def select_k_nearest_neighbors(points, center, k):
    """Return the k points in dataset nearest center."""
    if k > len(points):
        raise ValueError('k must be less than the size of the dataset')

    # Calculate distances, for use in selecting closest
    distances = [calculate.distance(point[0], center) for point in points]
    point_distances = zip(distances, points)

    # Find k points with smallest distances
    # NOTE: introselect algorithm is O(n) instead of O(n log k)
    # However, the only implementation in python requires converting to a
    # numpy array.
    # For indices: heapq.nsmallest(k, range(len(input_list)), key=input_list.__getitem__)
    nearest_distances = heapq.nsmallest(k, point_distances, key=lambda x: x[0])

    # Remove distance from tuples, leaving only the points
    _, sorted_points = zip(*nearest_distances)

    return sorted_points