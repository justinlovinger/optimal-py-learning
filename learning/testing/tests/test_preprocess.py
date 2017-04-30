import random
import copy

import pytest
import numpy

from pynn import preprocess
from pynn.data import datasets

from pynn.testing import helpers

def test_normalize():
    random_matrix = numpy.random.rand(random.randint(2, 10),
                                      random.randint(1, 10))
    print 'Generated Matrix:'
    print random_matrix

    normalized_matrix = preprocess.normalize(random_matrix)

    assert random_matrix.shape == normalized_matrix.shape

    # Original matrix should be unchanged
    assert not numpy.array_equal(random_matrix, normalized_matrix)
    
    # Normalized matrix should have mean of 0 standard deviation of 1
    # for each dimension
    means = numpy.mean(normalized_matrix, 0)
    for mean in means:
        print mean
        assert helpers.approx_equal(mean, 0, tol=1e-10)
    sds = numpy.std(normalized_matrix, 0)
    for sd in sds:
        print sd
        assert helpers.approx_equal(sd, 1, tol=1e-10)

    # TODO: deterministic test
    #inputs = [[0.75, 0.25],
    #          [0.5, 0.5],
    #          [0.25, 0.75]]
    #expected = [

    #assert preprocess.normalize(inputs) == numpy.matrix(expected)

def test_normalize_one_row():
    matrix = [[0, 1, 2]]
    with pytest.raises(ValueError):
        preprocess.normalize(matrix)
    with pytest.raises(ValueError):
        preprocess.normalize(numpy.array(matrix))


######################
# Depuration functions
######################
def test_list_minus_i():
    list_ = [0, 1, 2]
    assert preprocess._list_minus_i(list_, 0) == [1, 2]
    assert preprocess._list_minus_i(list_, 1) == [0, 2]
    assert preprocess._list_minus_i(list_, 2) == [0, 1]

def test_count_classes():
    dataset = datasets.get_xor()
    class_counts = preprocess._count_classes(dataset)

    assert len(class_counts) == 2
    assert class_counts[(0,)] == 2
    assert class_counts[(1,)] == 2

    dataset = [
         ([0], ['foo']),
         ([0], ['bar']),
         ([0], ['bar'])
        ]
    class_counts = preprocess._count_classes(dataset)

    assert len(class_counts) == 2
    assert class_counts[('foo',)] == 1
    assert class_counts[('bar',)] == 2

def test_clean_dataset_depuration():
    dataset = [
               ([0.0], (0,)),
               ([0.0], (0,)),
               ([0.0], (0,)),
               ([0.01], (1,)),
               ([0.5], (0.5,)),
               ([0.5], (0.5,)),
               ([0.99], (0,)),
               ([1.0], (1,)),
               ([1.0], (1,)),
               ([1.0], (1,)),
              ]

    cleaned_dataset, changed_points, removed_points = preprocess.clean_dataset_depuration(dataset, k=3, k_prime=2)
    assert cleaned_dataset == [
                               ([0.0], (0,)),
                               ([0.0], (0,)),
                               ([0.0], (0,)),
                               ([0.01], (0,)),
                               ([0.99], (1,)),
                               ([1.0], (1,)),
                               ([1.0], (1,)),
                               ([1.0], (1,)),
                              ]
    assert changed_points == [3, 6]
    assert removed_points == [4, 5]

######################
# PCA
######################
def test_pca_using_expected_num_dimensions():
    data = [[-1, -1], 
            [1, 1]]
    expected = numpy.matrix([[-1], [1]])

    assert numpy.array_equal(preprocess.pca(data, 1),
                             expected)

def test_pca_using_num_dimensions_func():
    data = [[-1, -1], 
            [1, 1]]
    expected = numpy.matrix([[-1], [1]])

    def selection_func(eigen_values):
        return [i for i, v in enumerate(eigen_values) if v > 1]

    assert numpy.array_equal(preprocess.pca(data,
                                            select_dimensions_func=selection_func),
                             expected)

def test_pca_no_expected_or_func():
    with pytest.raises(ValueError):
        preprocess.pca([], None, None)

def test_pca_both_expected_and_func():
    with pytest.raises(ValueError):
        preprocess.pca([], 1, lambda x: [0])

###########################
# Default cleaning function
###########################
def test_clean_dataset_no_pca():
    # Should just apply depuration
    dataset = [
               ([0.0], (0,)),
               ([0.0], (0,)),
               ([0.0], (0,)),
               ([0.0], (1,)),
               ([1.0], (0,)),
               ([1.0], (1,)),
               ([1.0], (1,)),
               ([1.0], (1,)),
              ]

    # Normalize input
    # And depuration should correct the 4th and 5th targets
    expected = [
                (numpy.array([-1.0]), (0,)),
                (numpy.array([-1.0]), (0,)),
                (numpy.array([-1.0]), (0,)),
                (numpy.array([-1.0]), (0,)),
                (numpy.array([1.0]), (1,)),
                (numpy.array([1.0]), (1,)),
                (numpy.array([1.0]), (1,)),
                (numpy.array([1.0]), (1,)),
               ]
    assert preprocess.clean_dataset(dataset) == expected

def test_clean_dataset_with_pca():
    dataset = [
               ([0.0, 0.0], (0,)),
               ([0.0, 0.0], (0,)),
               ([0.0, 0.0], (0,)),
               ([0.0, 0.0], (1,)),
               ([1.0, 1.0], (0,)),
               ([1.0, 1.0], (1,)),
               ([1.0, 1.0], (1,)),
               ([1.0, 1.0], (1,)),
              ]

    # PCA should reduce to one dimension (since it is currently just on a
    # diagonal)
    # PCA will also normalize input
    # And depuration should correct the 4th and 5th targets
    expected = [
                (numpy.array([-1.0]), (0,)),
                (numpy.array([-1.0]), (0,)),
                (numpy.array([-1.0]), (0,)),
                (numpy.array([-1.0]), (0,)),
                (numpy.array([1.0]), (1,)),
                (numpy.array([1.0]), (1,)),
                (numpy.array([1.0]), (1,)),
                (numpy.array([1.0]), (1,)),
               ]
    assert preprocess.clean_dataset(dataset) == expected