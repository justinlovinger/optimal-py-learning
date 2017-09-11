import random

from learning.data import datasets

def test_lenses():
    patterns = datasets.get_lenses()
    assert patterns[0] == ([-1, -1, -1, -1], [0, 0, 1])
    assert patterns[1] == ([-1, -1, -1, 1], [0, 1, 0])
    assert patterns[8] == ([0, -1, -1, -1], [0, 0, 1])

def test_get_random_classification_dataset():
    num_points = random.randint(1, 100)
    input_size = random.randint(1, 100)
    num_classes = random.randint(2, 10)

    dataset = datasets.get_random_classification(num_points, input_size, num_classes)

    assert len(dataset) == num_points
    for point in dataset:
        assert len(point) == 2 # (inputs, target) pair
        assert len(point[0]) == input_size
        assert len(point[1]) == num_classes

        # Check values in range
        for inputs in point[0]:
            assert 0 <= inputs <= 1

        # Target has a single 1.0
        target_count = 0
        for val in point[1]:
            if val == 1.0:
                target_count += 1
        assert target_count == 1

def test_get_random_regression_dataset():
    num_points = random.randint(1, 100)
    input_size = random.randint(1, 100)
    num_targets = random.randint(2, 10)

    dataset = datasets.get_random_regression(num_points, input_size, num_targets)

    assert len(dataset) == num_points
    for point in dataset:
        assert len(point) == 2 # (inputs, target) pair
        assert len(point[0]) == input_size
        assert len(point[1]) == num_targets

        # Check values in range
        for input_ in point[0]:
            assert 0 <= input_ <= 1

        # Target is a random vector
        for target_ in point[1]:
            assert 0 <= target_ <= 1

