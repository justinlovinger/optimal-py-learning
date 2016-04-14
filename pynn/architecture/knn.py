"""Layers and functions for a k-nearest-neighbors architecture."""

from pynn import calculate

def select_k_nearest_neighbors(points, center, k):
    """Return the k points in dataset nearest center."""
    if k > len(points):
        raise ValueError('k must be less than the size of the dataset')

    # TODO: more efficient implementation, without sorting
    # Sort each point by distance to center
    distances = [calculate.distance(point[0], center) for point in points]
    point_distances = zip(points, distances)
    point_distances.sort(key=lambda x: x[1])
    sorted_points, _ = zip(*point_distances)

    # Select the k closest, using our sorted list
    return sorted_points[:k]