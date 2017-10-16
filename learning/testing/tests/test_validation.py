import random
import time

import numpy

from learning import validation, error
from learning.data import datasets

from learning.testing import helpers


#################
# Cross validate
#################
def test_validate_network(monkeypatch):
    # Patch time.clock so time attribute is deterministic
    monkeypatch.setattr(time, 'clock', lambda: 0.0)

    # This lets us test training error, but is kinda complicated
    model = helpers.WeightedSumModel()

    assert (helpers.fix_numpy_array_equality(
        validation._validate_model(model, (numpy.array([[1], [1]]), numpy.array([[0], [1]])),
                                   (numpy.array([[1]]), numpy.array([[2]])),
                                   iterations=0, _classification=True))
            == helpers.fix_numpy_array_equality(
                {'time': 0.0, 'epochs': 0,
                 'training_error': 0.5,
                 'testing_error': 1.0,
                 'training_accuracy': 0.5,
                 'training_confusion_matrix': numpy.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]),
                 'testing_accuracy': 0.0,
                 'testing_confusion_matrix': numpy.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])}))

    # Make network that returns set output for a given input
    # Simpler, always 0 training error
    model = helpers.RememberPatternsModel()
    assert (validation._validate_model(model, (numpy.array([[1]]), numpy.array([[1]])),
                                       (numpy.array([[1]]), numpy.array([[1.5]])),
                                       iterations=0, _classification=False)
            == {'time': 0.0, 'epochs': 0,
                'training_error': 0.0,
                'testing_error': 0.25})
    assert (validation._validate_model(model, (numpy.array([[0]]), numpy.array([[0]])),
                                       (numpy.array([[0]]), numpy.array([[2.0]])),
                                       iterations=0, _classification=False)
            == {'time': 0.0, 'epochs': 0,
                'training_error': 0.0,
                'testing_error': 4.0})


def test_cross_validate(monkeypatch):
    # Patch time.clock so time attribute is deterministic
    monkeypatch.setattr(time, 'clock', lambda : 0.0)

    # Make network that returns set output for
    patterns = [
        ([0], [1]),
        ([1], [1]),
        ([2], [1])
    ]
    model = helpers.SetOutputModel([1])

    # Track patterns for training
    training_patterns = []
    def post_pattern_callback(network_, input_vec, target_vec):
        training_patterns.append((list(input_vec), list(target_vec)))

    # Cross validate with deterministic network, and check output
    stats = validation.cross_validate(model, zip(*patterns), num_folds=3,
                                      iterations=1,
                                      post_pattern_callback=post_pattern_callback)

    # Check
    assert (helpers.fix_numpy_array_equality(stats)
            == helpers.fix_numpy_array_equality(_CROSS_VALIDATION_STATS))
    assert training_patterns == [([1], [1]), ([2], [1]), # First fold
                                 ([0], [1]), ([2], [1]), # Second fold
                                 ([0], [1]), ([1], [1])] # Third fold


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
    model = helpers.SetOutputModel([1])

    # Track patterns for training
    training_patterns = []
    def post_pattern_callback(network_, input_vec, target_vec):
        training_patterns.append((list(input_vec), list(target_vec)))

    # Cross validate with deterministic network, and check output
    stats = validation.benchmark(model, zip(*patterns), num_folds=3, num_runs=2,
                                 iterations=1,
                                 post_pattern_callback=post_pattern_callback)

    # Check
    assert (helpers.fix_numpy_array_equality(stats)
            == helpers.fix_numpy_array_equality(_BENCHMARK_STATS))
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
    model = helpers.SetOutputModel([1])
    model2 = helpers.SetOutputModel([1])

    # Cross validate with deterministic network, and check output
    stats = validation.compare(['model', 'model2'], [model, model2], zip(*patterns),
                               num_folds=3, num_runs=2, all_kwargs={'iterations':1})

    # Check
    assert (helpers.fix_numpy_array_equality(stats)
            == helpers.fix_numpy_array_equality(_COMPARE_STATS))


_VALIDATION_STATS = {'time': 0.0, 'epochs': 1,
                     'training_error': 0.0, 'testing_error': 0.0,
                     'training_accuracy': 1.0,
                     'training_confusion_matrix': numpy.array([[0, 0], [0, 2]]),
                     'testing_accuracy': 1.0,
                     'testing_confusion_matrix': numpy.array([[0, 0], [0, 1]])}

_CROSS_VALIDATION_STATS = {'folds': [_VALIDATION_STATS, _VALIDATION_STATS,
                                     _VALIDATION_STATS],
                           'mean': {'time': 0.0, 'epochs': 1,
                                    'training_error': 0.0, 'testing_error': 0.0,
                                    'training_accuracy': 1.0,
                                    'training_confusion_matrix': numpy.array([[0, 0], [0, 2]]),
                                    'testing_accuracy': 1.0,
                                    'testing_confusion_matrix': numpy.array([[0, 0], [0, 1]])},
                           'sd': {'time': 0.0, 'epochs': 0.0,
                                  'training_error': 0.0, 'testing_error': 0.0,
                                  'training_accuracy': 0.0,
                                  'training_confusion_matrix': numpy.array([[0, 0], [0, 0]]),
                                  'testing_accuracy': 0.0,
                                  'testing_confusion_matrix': numpy.array([[0, 0], [0, 0]])}}

_BENCHMARK_STATS = {'runs': [_CROSS_VALIDATION_STATS, _CROSS_VALIDATION_STATS],
                    'mean_of_means': {'time': 0.0, 'epochs': 1,
                                      'training_error': 0.0, 'testing_error': 0.0,
                                      'training_accuracy': 1.0,
                                      'training_confusion_matrix': numpy.array([[0, 0], [0, 2]]),
                                      'testing_accuracy': 1.0,
                                      'testing_confusion_matrix': numpy.array([[0, 0], [0, 1]])},
                    'sd_of_means': {'time': 0.0, 'epochs': 0.0,
                                    'training_error': 0.0, 'testing_error': 0.0,
                                    'training_accuracy': 0.0,
                                    'training_confusion_matrix': numpy.array([[0, 0], [0, 0]]),
                                    'testing_accuracy': 0.0,
                                    'testing_confusion_matrix': numpy.array([[0, 0], [0, 0]])}
                   }

_COMPARE_STATS = {'model': _BENCHMARK_STATS,
                  'model2':_BENCHMARK_STATS,
                  'mean_of_means': {'time': 0.0, 'epochs': 1,
                                    'training_error': 0.0, 'testing_error': 0.0,
                                    'training_accuracy': 1.0,
                                    'training_confusion_matrix': numpy.array([[0, 0], [0, 2]]),
                                    'testing_accuracy': 1.0,
                                    'testing_confusion_matrix': numpy.array([[0, 0], [0, 1]])},
                  'sd_of_means': {'time': 0.0, 'epochs': 0.0,
                                  'training_error': 0.0, 'testing_error': 0.0,
                                  'training_accuracy': 0.0,
                                  'training_confusion_matrix': numpy.array([[0, 0], [0, 0]]),
                                  'testing_accuracy': 0.0,
                                  'testing_confusion_matrix': numpy.array([[0, 0], [0, 0]])}
                 }


def test_isdataset():
    assert validation._isdataset(datasets.get_xor()) is True
    assert validation._isdataset([datasets.get_and(), datasets.get_xor()]) is False


#################
# Metrics
#################
def test_get_error():
    model = helpers.SetOutputModel([1])
    assert validation.get_error(
        model, numpy.array([[1]]), numpy.array([[0]]),
        error_func=error.MSE()) == 1.0
    assert validation.get_error(
        model, numpy.array([[1]]), numpy.array([[1]]),
        error_func=error.MSE()) == 0.0
    assert validation.get_error(
        model, numpy.array([[1]]), numpy.array([[0.5]]),
        error_func=error.MSE()) == 0.25
    assert validation.get_error(
        model,
        numpy.array([[1], [1]]),
        numpy.array([[1], [0]]),
        error_func=error.MSE()) == 0.5
    assert validation.get_error(
        model,
        numpy.array([[1], [1]]),
        numpy.array([[0.5], [0.5]]),
        error_func=error.MSE()) == 0.25


def test_get_accuracy():
    model = helpers.SetOutputModel([1])
    assert validation.get_accuracy(model,
                                   numpy.array([[1], [1]]),
                                   numpy.array([[1], [0]])) == 0.5
    assert validation.get_accuracy(model,
                                   numpy.array([[1], [1]]),
                                   numpy.array([[1], [1]])) == 1.0
    assert validation.get_accuracy(model,
                                   numpy.array([[1], [1]]),
                                   numpy.array([[0], [0]])) == 0.0


def test__get_accuracy():
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
        numpy.array([0, 0, 1, 1]),
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

    assert (validation._get_confusion_matrix(
        numpy.array([0, 0, 1, 1, 0, 1, 1, 0]),
        numpy.array([0, 1, 0, 1, 0, 0, 0, 1]),
        2)
            == numpy.array([[2, 3],
                            [2, 1]])).all()


def test_get_confusion_matrix_1_class():
    actual = numpy.array([0, 0])
    expected = numpy.array([0, 0])
    assert (validation._get_confusion_matrix(actual, expected, 1)
            == numpy.array([[2]])).all()


def test_get_classes_matrix():
    assert (validation._get_classes(
        numpy.array([[1.0, 0.75, 0.25, 0.0],
                     [-1.0, -0.75, -0.25, 0.0]]))
            == numpy.array([0, 3])).all()


def test_get_classes_label_vec():
    assert (validation._get_classes(
        numpy.array([[1], [0], ['foo']])) == numpy.array([1, 0, 'foo'])).all()


############################
# Splitting datasets
############################
def test_make_train_test_sets_1d_labels():
    inputs = numpy.array([[0.0, 0.0],
                          [1.0, 0.0],
                          [0.0, 1.0],
                          [1.0, 1.0],
                          [0.0, 1.0],
                          [1.0, 1.0]])
    labels = numpy.array([[0],
                          [1],
                          [1],
                          [0],
                          [1],
                          [0]])

    assert (helpers.fix_numpy_array_equality(
        validation.make_train_test_sets(inputs, labels, 1))
            == helpers.fix_numpy_array_equality(
                ((inputs[:2], labels[:2]), (inputs[2:], labels[2:]))))


def test_make_train_test_sets_2d_labels():
    inputs = numpy.array([[0.0, 0.0],
                          [1.0, 0.0],
                          [0.0, 1.0],
                          [1.0, 1.0],
                          [0.0, 1.0],
                          [1.0, 1.0]])
    labels = numpy.array([[0, 1],
                          [1, 0],
                          [1, 0],
                          [0, 1],
                          [1, 0],
                          [0, 1]])

    assert (helpers.fix_numpy_array_equality(
        validation.make_train_test_sets(inputs, labels, 1))
            == helpers.fix_numpy_array_equality(
                ((inputs[:2], labels[:2]), (inputs[2:], labels[2:]))))


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


#############################
# Statistics
#############################
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
