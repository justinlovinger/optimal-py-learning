import random
import time

from pynn import validation
from pynn import network
from pynn.testing import helpers
from pynn.architecture import pbnn

def random_dataset():
    dataset = []
    inputs = random.randint(2, 5)
    targets = random.randint(1, 3)
    # Dataset of size between 100 - 150
    for i in range(random.randint(100, 150)):
        input = []
        for j in range(inputs):
            input.append(random.uniform(-1.0, 1.0))
        target = []
        for j in range(targets):
            target.append(random.uniform(-1.0, 1.0))
        
        # Each datapoint is an input, target pair
        dataset.append([input, target])

    return dataset

def test_split_dataset():
    dataset = random_dataset()
    num_sets = random.randint(2, 5)
    sets = validation._split_dataset(dataset, num_sets)

    # The right number of sets is created
    assert len(sets) == num_sets

    for i in range(num_sets):
        for j in range(i+1, num_sets):
            # Check that each set is about equal in size
            assert len(sets[i]) >= len(sets[j])-5 and len(sets[i]) <= len(sets[j])+5

            # Check that each set has unique datapoints
            for point1 in sets[i]:
                for point2 in sets[j]:
                    assert point1 != point2

def test_mean_of_dicts():
    folds = [{'test': 0.0, 'test2': 1.0},
             {'test': 1.0, 'test2': 2.0}]
    assert validation._mean_of_dicts(folds) == {'test': 0.5, 'test2': 1.5}

def test_sd_of_dicts():
    folds = [{'test': 0.0, 'test2': 1.0},
             {'test': 1.0, 'test2': 2.0}]
    means = validation._mean_of_dicts(folds)
    assert validation._sd_of_dicts(folds, means) == {'test': 0.5,
                                                     'test2': 0.5}

def test_validate_network(monkeypatch):
    # Patch time.clock so time attribute is deterministic
    monkeypatch.setattr(time, 'clock', lambda : 0.0)

    # Make network that returns set output for a given input
    patterns = [
                ([1], [0]),
                ([1], [1]),
                ([1], [2])
               ]
    store_targets = pbnn.StoreTargetsLayer()
    summation = pbnn.WeightedSummationLayer()
    nn = network.Network({'I': [summation],
                          store_targets: [summation],
                          summation: ['O']},
                         incoming_order_dict = {summation: ['I', store_targets]})

    assert validation._validate_network(nn, [([1], [0]), ([1], [1])],
                                        [([1], [2])],
                                        iterations=0) == {'time': 0.0, 'epochs': 0,
                                                          'training_error': 0.5,
                                                          'testing_error': 1.0}

def test_cross_validate(monkeypatch):
    # Make network that returns set output for
    patterns = [
                ([0], [1]),
                ([1], [1]),
                ([2], [1])
               ]
    nn = network.Network([helpers.SetOutputLayer([1])])

    # Cross validate with deterministic network, and check output
    
    # Patch time.clock so time attribute is deterministic
    monkeypatch.setattr(time, 'clock', lambda : 0.0)

    # Track patterns for training
    training_patterns = []
    def post_pattern_callback(network_, pattern):
        training_patterns.append(pattern)

    # Validate
    stats = validation.cross_validate(nn, patterns, num_folds=3,
                                      iterations=1,
                                      post_pattern_callback=post_pattern_callback)

    # Check
    assert stats == {'folds': [{'time': 0.0, 'epochs': 1,
                                'training_error': 0.0, 'testing_error': 0.0},
                               {'time': 0.0, 'epochs': 1,
                                'training_error': 0.0, 'testing_error': 0.0},
                               {'time': 0.0, 'epochs': 1,
                                'training_error': 0.0, 'testing_error': 0.0}],
                     'mean': {'time': 0.0, 'epochs': 1,
                              'training_error': 0.0, 'testing_error': 0.0},
                     'sd': {'time': 0.0, 'epochs': 0.0,
                            'training_error': 0.0, 'testing_error': 0.0}}
    assert training_patterns == [([1], [1]), ([2], [1]), # First fold
                                 ([0], [1]), ([2], [1]), # Second fold
                                 ([0], [1]), ([1], [1])] # Third fold