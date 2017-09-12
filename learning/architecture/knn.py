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
