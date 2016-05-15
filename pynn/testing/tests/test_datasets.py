import random

from pynn.data import datasets

def test_lenses():
    patterns = datasets.get_lenses()
    assert patterns[0] == ([-1, -1, -1, -1], [0, 0, 1])
    assert patterns[1] == ([-1, -1, -1, 1], [0, 1, 0])
    assert patterns[8] == ([0, -1, -1, -1], [0, 0, 1])

def test_get_random_dataset():
    num_points = random.randint(1, 100)
    input_size = random.randint(1, 100)

    dataset = datasets.get_random(num_points, input_size)
    
    assert len(dataset) == num_points
    for point in dataset:
        assert len(point) == 2 # (inputs, target) pair
        assert len(point[0]) == input_size
        assert len(point[1]) == 1 # Currently hardcoded to 1 target value

        # Check values in range
        for input in point[0]:
            assert 0 <= input <= 1
        assert 0 <= point[1][0] <= 1