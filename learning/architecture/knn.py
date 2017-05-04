"""Layers and functions for a k-nearest-neighbors architecture."""
import heapq

from learning import calculate

def select_k_nearest_neighbors(matrix, center, k):
    """Return the k indexes of rows in matrix nearest center."""
    if k > len(matrix):
        raise ValueError('k must be less than the rows in the matrix')

    # Calculate distances, for use in selecting closest
    # TODO: calculate.distance should be able to take whole matrix
    distances = [calculate.distance(vec, center) for vec in matrix]

    # Find k vectors with smallest distances
    # NOTE: Introselect algorithm, is O(n) instead of O(n log k)
    #   However, the only implementation in python requires converting to a
    #   numpy array.
    nearest_indices = heapq.nsmallest(k, range(len(distances)), key=distances.__getitem__)

    return nearest_indices
