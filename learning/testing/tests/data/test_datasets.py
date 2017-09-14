import random

import numpy

from learning.data import datasets

def test_lenses():
    input_matrix, target_matrix = datasets.get_lenses()

    assert (input_matrix[0] == numpy.array([-1, -1, -1, -1])).all()
    assert (target_matrix[0] == numpy.array([0, 0, 1])).all()

    assert (input_matrix[1] == numpy.array([-1, -1, -1, 1])).all()
    assert (target_matrix[1] == numpy.array([0, 1, 0])).all()

    assert (input_matrix[8] == numpy.array([0, -1, -1, -1])).all()
    assert (target_matrix[8] == numpy.array([0, 0, 1])).all()


def test_get_random_classification_dataset():
    num_points = random.randint(1, 100)
    input_size = random.randint(1, 100)
    num_classes = random.randint(2, 10)

    input_matrix, target_matrix = datasets.get_random_classification(
        num_points, input_size, num_classes)

    assert len(input_matrix) == num_points
    assert len(target_matrix) == num_points

    for inp_vec, tar_vec in zip(input_matrix, target_matrix):
        assert len(inp_vec) == input_size
        assert len(tar_vec) == num_classes

        # Check values in range
        for val in inp_vec:
            assert -1 <= val <= 1

        # Target has a single 1.0
        target_count = 0
        for val in tar_vec:
            if val == 1.0:
                target_count += 1
        assert target_count == 1


def test_get_random_regression_dataset():
    num_points = random.randint(1, 100)
    input_size = random.randint(1, 100)
    num_targets = random.randint(2, 10)

    input_matrix, target_matrix = datasets.get_random_regression(
        num_points, input_size, num_targets)

    assert len(input_matrix) == num_points
    assert len(target_matrix) == num_points

    for inp_vec, tar_vec in zip(input_matrix, target_matrix):
        assert len(inp_vec) == input_size
        assert len(tar_vec) == num_targets

        # Check values in range
        for val in inp_vec:
            assert -1 <= val <= 1

        # Target is a random vector
        for val in tar_vec:
            assert -1 <= val <= 1
