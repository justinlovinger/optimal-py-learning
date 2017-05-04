import random
import time

import numpy

from learning import validation
from learning.data import datasets

from learning.testing import helpers

def test_split_dataset():
    input_matrix, target_matrix = datasets.get_random_regression(
        random.randint(100, 150), random.randint(2, 5), random.randint(1, 3))
    num_sets = random.randint(2, 5)
    sets = validation._split_dataset(input_matrix, target_matrix, num_sets)

    # The right number of sets is created
    assert len(sets) == num_sets

    for i in range(num_sets):
        for j in range(i+1, num_sets):
            # Check that each set is about equal in size
            assert len(sets[i]) >= len(sets[j])-5 and len(sets[i]) <= len(sets[j])+5

            # Check that each set has unique patterns
            patterns = zip(*sets[i])
            other_patterns = zip(*sets[j])

            for pattern in patterns:
                for other_pattern in other_patterns:
                    assert not ((pattern[0] == other_pattern[0]).all()
                                and (pattern[1] == other_pattern[1]).all())

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

#################
# Cross validate
#################
def test_validate_network(monkeypatch):
    # Patch time.clock so time attribute is deterministic
    monkeypatch.setattr(time, 'clock', lambda: 0.0)

    # This lets us test training error, but is kinda complicated
    nn = helpers.WeightedSumModel()

    assert (validation._validate_model(nn, (numpy.array([[1], [1]]), numpy.array([[0], [1]])),
                                       (numpy.array([[1]]), numpy.array([[2]])),
                                       iterations=0)
            == {'time': 0.0, 'epochs': 0,
                'training_error': 0.5,
                'testing_error': 1.0,
                'training_accuracy': 1.0,
                'training_confusion_matrix': numpy.array([[2]]),
                'testing_accuracy': 1.0,
                'testing_confusion_matrix': numpy.array([[1]])})

    # Make network that returns set output for a given input
    # Simpler, always 0 training error
    nn = helpers.RememberPatternsModel()
    assert (validation._validate_model(nn, (numpy.array([[1]]), numpy.array([[1]])),
                                       (numpy.array([[1]]), numpy.array([[1.5]])),
                                       iterations=0)
            == {'time': 0.0, 'epochs': 0,
                'training_error': 0.0,
                'testing_error': 0.25,
                'training_accuracy': 1.0,
                'training_confusion_matrix': numpy.array([[1]]),
                'testing_accuracy': 1.0,
                'testing_confusion_matrix': numpy.array([[1]])})
    assert (validation._validate_model(nn, (numpy.array([[0]]), numpy.array([[0]])),
                                       (numpy.array([[0]]), numpy.array([[2.0]])),
                                       iterations=0)
            == {'time': 0.0, 'epochs': 0,
                'training_error': 0.0,
                'testing_error': 4.0,
                'training_accuracy': 1.0,
                'training_confusion_matrix': numpy.array([[1]]),
                'testing_accuracy': 1.0,
                'testing_confusion_matrix': numpy.array([[1]])})

def test_cross_validate(monkeypatch):
    # Patch time.clock so time attribute is deterministic
    monkeypatch.setattr(time, 'clock', lambda : 0.0)

    # Make network that returns set output for
    patterns = [
        ([0], [1]),
        ([1], [1]),
        ([2], [1])
    ]
    nn = helpers.SetOutputModel([1])

    # Track patterns for training
    training_patterns = []
    def post_pattern_callback(network_, input_vec, target_vec):
        training_patterns.append((list(input_vec), list(target_vec)))

    # Cross validate with deterministic network, and check output
    stats = validation.cross_validate(nn, zip(*patterns), num_folds=3,
                                      iterations=1,
                                      post_pattern_callback=post_pattern_callback)

    # Check
    assert stats == _CROSS_VALIDATION_STATS
    assert training_patterns == [([1], [1]), ([2], [1]), # First fold
                                 ([0], [1]), ([2], [1]), # Second fold
                                 ([0], [1]), ([1], [1])] # Third fold

################
# Stat functions
################
def test_get_classes():
    assert (validation._get_classes(
        numpy.array([[1.0, 0.75, 0.25, 0.0],
                     [-1.0, -0.75, -0.25, 0.0]]))
            == numpy.array([0, 3])).all()

def test_get_accuracy():
    assert validation._get_accuracy(
        numpy.array([0, 1, 2, 3]),
        numpy.array([1, 0, 0, 0])) == 0.0

    assert validation._get_accuracy(
        numpy.array([0, 1, 2, 3]),
        numpy.array([0, 0, 0, 0])) == 0.25

    assert validation._get_accuracy(
        numpy.array([0, 1, 2, 3]),
        numpy.array([0, 1, 0, 0])) == 0.5

    assert validation._get_accuracy(
        numpy.array([0, 1, 2, 3]),
        numpy.array([0, 1, 2, 0])) == 0.75

    assert validation._get_accuracy(
        numpy.array([0, 1, 2, 3]),
        numpy.array([0, 1, 2, 3])) == 1.0


def test_get_confusion_matrix():
    assert (validation._get_confusion_matrix(
        numpy.array([0, 1, 1, 0]),
        numpy.array([0, 1, 0, 1]),
        2)
            == numpy.array([[1, 1],
                            [1, 1]])).all()

    assert (validation._get_confusion_matrix(
        numpy.array([1, 1, 1, 1]),
        numpy.array([0, 0, 0, 0]),
        2)
            == numpy.array([[0, 4],
                            [0, 0]])).all()

################
# Benchmark
################
def test_benchmark(monkeypatch):
    # Patch time.clock so time attribute is deterministic
    monkeypatch.setattr(time, 'clock', lambda : 0.0)

    # Make network that returns set output for
    patterns = [
        ([0], [1]),
        ([1], [1]),
        ([2], [1])
    ]
    nn = helpers.SetOutputModel([1])

    # Track patterns for training
    training_patterns = []
    def post_pattern_callback(network_, input_vec, target_vec):
        training_patterns.append((list(input_vec), list(target_vec)))

    # Cross validate with deterministic network, and check output
    stats = validation.benchmark(nn, zip(*patterns), num_folds=3, num_runs=2,
                                 iterations=1,
                                 post_pattern_callback=post_pattern_callback)

    # Check
    assert stats == _BENCHMARK_STATS
    assert training_patterns == [([1], [1]), ([2], [1]), # First fold
                                 ([0], [1]), ([2], [1]), # Second fold
                                 ([0], [1]), ([1], [1]), # Third fold
                                 ([1], [1]), ([2], [1]), # First fold 2
                                 ([0], [1]), ([2], [1]), # Second fold 2
                                 ([0], [1]), ([1], [1])] # Third fold 2

####################
# Compare
####################
def test_compare(monkeypatch):
    # Patch time.clock so time attribute is deterministic
    monkeypatch.setattr(time, 'clock', lambda : 0.0)

    # Make network that returns set output for
    patterns = [
        ([0], [1]),
        ([1], [1]),
        ([2], [1])
    ]
    nn = helpers.SetOutputModel([1])
    nn2 = helpers.SetOutputModel([1])

    # Cross validate with deterministic network, and check output
    stats = validation.compare([('nn', nn, zip(*patterns), {'iterations':1}),
                                ('nn2', nn2, zip(*patterns), {'iterations':1})],
                               num_folds=3, num_runs=2)

    # Check
    assert stats == {'nn': _BENCHMARK_STATS,
                     'nn2':_BENCHMARK_STATS,
                     'mean_of_means': {'time': 0.0, 'epochs': 1,
                                       'training_error': 0.0, 'testing_error': 0.0,
                                       'training_accuracy': 1.0,
                                       'training_confusion_matrix': numpy.array([[2]]),
                                       'testing_accuracy': 1.0,
                                       'testing_confusion_matrix': numpy.array([[1]])},
                     'sd_of_means': {'time': 0.0, 'epochs': 0.0,
                                     'training_error': 0.0, 'testing_error': 0.0,
                                     'training_accuracy': 0.0,
                                     'training_confusion_matrix': numpy.array([[0]]),
                                     'testing_accuracy': 0.0,
                                     'testing_confusion_matrix': numpy.array([[0]])}
                    }


_VALIDATION_STATS = {'time': 0.0, 'epochs': 1,
                     'training_error': 0.0, 'testing_error': 0.0,
                     'training_accuracy': 1.0,
                     'training_confusion_matrix': numpy.array([[2]]),
                     'testing_accuracy': 1.0,
                     'testing_confusion_matrix': numpy.array([[1]])}

_CROSS_VALIDATION_STATS = {'folds': [_VALIDATION_STATS, _VALIDATION_STATS,
                                     _VALIDATION_STATS],
                           'mean': {'time': 0.0, 'epochs': 1,
                                    'training_error': 0.0, 'testing_error': 0.0,
                                    'training_accuracy': 1.0,
                                    'training_confusion_matrix': numpy.array([[2]]),
                                    'testing_accuracy': 1.0,
                                    'testing_confusion_matrix': numpy.array([[1]])},
                           'sd': {'time': 0.0, 'epochs': 0.0,
                                  'training_error': 0.0, 'testing_error': 0.0,
                                  'training_accuracy': 0.0,
                                  'training_confusion_matrix': numpy.array([[0]]),
                                  'testing_accuracy': 0.0,
                                  'testing_confusion_matrix': numpy.array([[0]])}}

_BENCHMARK_STATS = {'runs': [_CROSS_VALIDATION_STATS, _CROSS_VALIDATION_STATS],
                    'mean_of_means': {'time': 0.0, 'epochs': 1,
                                      'training_error': 0.0, 'testing_error': 0.0,
                                      'training_accuracy': 1.0,
                                      'training_confusion_matrix': numpy.array([[2]]),
                                      'testing_accuracy': 1.0,
                                      'testing_confusion_matrix': numpy.array([[1]])},
                    'sd_of_means': {'time': 0.0, 'epochs': 0.0,
                                    'training_error': 0.0, 'testing_error': 0.0,
                                    'training_accuracy': 0.0,
                                    'training_confusion_matrix': numpy.array([[0]]),
                                    'testing_accuracy': 0.0,
                                    'testing_confusion_matrix': numpy.array([[0]])}
                   }
