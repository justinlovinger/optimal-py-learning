import random
import copy

import pytest
import numpy

from learning import preprocess
from learning.data import datasets

from learning.testing import helpers

def test_shuffle_dataset_does_shuffle():
    dataset = (
        numpy.array(
            [['s1i1', 's1i2'],
             ['s2i1', 's2i2'],
             ['s3i1', 's3i2']]
        ),
        numpy.array(
            [['s1t1'],
             ['s2t1'],
             ['s3t1']]
        )
    )

    # Assert that not all shuffled sets match
    def _eq_dataset(dataset_a, dataset_b):
        return (dataset_a[0] == dataset_b[0]).all() and (dataset_a[1] == dataset_b[1]).all()
    shuffled_dataset = preprocess.shuffle(dataset)
    for i in range(20):
        if not _eq_dataset(shuffled_dataset, preprocess.shuffle(dataset)):
            return # They don't all match
    assert False, 'Add shuffled sets match'

def test_shuffle_dataset_correct_patterns():
    dataset = (
        numpy.array(
            [['s1i1', 's1i2'],
             ['s2i1', 's2i2'],
             ['s3i1', 's3i2']]
        ),
        numpy.array(
            [['s1t1'],
             ['s2t1'],
             ['s3t1']]
        )
    )

    shuffled_dataset = preprocess.shuffle(dataset)

    # Make mapping for testing shuffled set testing
    target_mapping = {tuple(inp_vec): tar_vec for inp_vec, tar_vec in zip(*dataset)}
    for inp_vec, tar_vec in zip(*shuffled_dataset):
        assert (target_mapping[tuple(inp_vec)] == tar_vec).all()
        target_mapping.pop(tuple(inp_vec))
    assert target_mapping == {} # Each original pattern is in shuffle dataset


def test_make_onehot_1d():
    assert (preprocess.make_onehot([1, 2, 1])
            == numpy.array([[1, 0], [0, 1], [1, 0]])).all()

    assert (preprocess.make_onehot([1, 2, 3, 1])
            == numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])).all()

def test_make_onehot_2d():
    labels = [
        [1, 2, 3],
        [2, 3, 4],
        [1, 2, 3]
    ]
    assert (preprocess.make_onehot(labels)
            == numpy.array([[1, 0], [0, 1], [1, 0]])).all()

#################
# Normalization
#################
def test_rescale():
    assert (preprocess.rescale(
        numpy.array([
            [-100, 2],
            [100, 0],
            [0, 1]
        ])
    ) == numpy.array([
        [-1.0, 1.0],
        [1.0, -1.0],
        [0.0, 0.0]
    ])).all()

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
    _, target_matrix = datasets.get_xor()
    class_counts = preprocess._count_classes(target_matrix)

    assert len(class_counts) == 2
    assert class_counts[(0, 1)] == 2
    assert class_counts[(1, 0)] == 2

    target_matrix = [
        ['foo'],
        ['bar'],
        ['bar']
    ]
    class_counts = preprocess._count_classes(target_matrix)

    assert len(class_counts) == 2
    assert class_counts[('foo',)] == 1
    assert class_counts[('bar',)] == 2

def test_clean_dataset_depuration():
    dataset = [
        [
            [0.0],
            [0.0],
            [0.0],
            [0.01],
            [0.5],
            [0.5],
            [0.99],
            [1.0],
            [1.0],
            [1.0],
        ],
        [
            (0,),
            (0,),
            (0,),
            (1,),
            (0.5,),
            (0.5,),
            (0,),
            (1,),
            (1,),
            (1,),
        ]
    ]

    cleaned_dataset, changed_points, removed_points = preprocess.clean_dataset_depuration(
        *dataset, k=3, k_prime=2)
    assert (numpy.array(cleaned_dataset) == numpy.array([
        [
            [0.0],
            [0.0],
            [0.0],
            [0.01],
            [0.99],
            [1.0],
            [1.0],
            [1.0],
        ],
        [
            (0,),
            (0,),
            (0,),
            (0,),
            (1,),
            (1,),
            (1,),
            (1,),
        ]
    ])).all()

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
    patterns = [
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
    assert ((numpy.array(preprocess.clean_dataset(*zip(*patterns)))
             == numpy.array(zip(*expected)))).all()

def test_clean_dataset_with_pca():
    patterns = [
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
    assert ((numpy.array(preprocess.clean_dataset(*zip(*patterns)))
             == numpy.array(zip(*expected)))).all()
