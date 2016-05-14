import random
import copy

import pytest
import numpy

from pynn import preprocess
from pynn.data import datasets

from pynn.testing import helpers

def test_normalize():
    random_matrix = numpy.random.rand(random.randint(1, 10),
                                      random.randint(1, 10))
    normalized_matrix = preprocess.normalize(random_matrix)

    assert random_matrix.shape == normalized_matrix.shape

    # Original matrix should be unchanged
    assert not numpy.array_equal(random_matrix, normalized_matrix)
    
    # Normalized matrix should have mean of 0 standard deviation of 1
    # for each dimension
    means = numpy.mean(normalized_matrix, 0)
    for mean in means:
        assert helpers.approx_equal(mean, 0, tol=1e-10)
    sds = numpy.std(normalized_matrix, 0)
    for sd in sds:
        assert helpers.approx_equal(sd, 1, tol=1e-10)

    # TODO: deterministic test
    #inputs = [[0.75, 0.25],
    #          [0.5, 0.5],
    #          [0.25, 0.75]]
    #expected = [

    #assert preprocess.normalize(inputs) == numpy.matrix(expected)


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

def test_clean_dataset():
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

    expected, _, _ = preprocess.clean_dataset_depuration(dataset)
    assert preprocess.clean_dataset(dataset) == expected

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
        return len([v for v in eigen_values if v > 1])

    assert numpy.array_equal(preprocess.pca(data,
                                            num_dimensions_func=selection_func),
                             expected)

def test_pca_no_expected_or_func():
    with pytest.raises(ValueError):
        preprocess.pca([], None, None)

def test_pca_both_expected_and_func():
    with pytest.raises(ValueError):
        preprocess.pca([], 1, lambda x: 1)