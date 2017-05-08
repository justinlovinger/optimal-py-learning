import math
import time
import numbers
import copy
import logging

import numpy

def train_test_validate(model, dataset, train_per_class, **kwargs):
    """Validate a classification dataset by splitting into a train and test set.

    Args:
        model: The model to validate.
        dataset: (input_matrix, label_matrix) tuple.
        train_per_class: Number of samples for each class in training set.
        **kwargs: Args passed to model.train
    """
    training_set, testing_set = _make_train_test_sets(*dataset, train_per_class=train_per_class)
    return _validate_model(model, training_set, testing_set, _classification=True, **kwargs)

def _make_train_test_sets(input_matrix, label_matrix, train_per_class):
    """Return ((training_inputs, training_labels), (testing_inputs, testing_labels)).

    Args:
        input_matrix: attributes matrix. Each row is sample, each column is attribute.
        label_matrix: labels matrix. Each row is sample, each column is label.
        train_per_class: Number of samples for each class in training set.
    """
    training_inputs = []
    training_labels = []
    testing_inputs = []
    testing_labels = []
    label_counts = {}

    # Add each row to training or testing set depending on count of labels
    for input_, label in zip(input_matrix, label_matrix):
        key = tuple(label)
        try:
            count = label_counts[key]
        except KeyError:
            # First time seeing label, count is 0
            count = 0

        if count < train_per_class:
            # Still need more training samples for this label
            training_inputs.append(input_)
            training_labels.append(label)
        else:
            # We have enough training samples for this label,
            # add to testing set instead
            testing_inputs.append(input_)
            testing_labels.append(label)

        label_counts[key] = count + 1

    return ((numpy.array(training_inputs), numpy.array(training_labels)),
            (numpy.array(testing_inputs), numpy.array(testing_labels)))

def _validate_model(model_, training_set, testing_set, _classification=True, **kwargs):
    """Test the given network on a partitular training and testing set."""
    # Train network on training set
    try:
        model_ = copy.deepcopy(model_) # No side effects
    except TypeError:
        logging.warning('Cannot pickle %s model.' \
                        'Cross validation may have side effects', type(model_))
    model_.reset()

    start = time.clock()
    model_.train(*training_set, **kwargs) # Train
    elapsed = time.clock() - start

    # Collect stats
    stats = {}
    stats['time'] = elapsed
    stats['epochs'] = model_.iteration

    # Get error for training and testing set
    stats['training_error'] = model_.avg_mse(*training_set)
    stats['testing_error'] = model_.avg_mse(*testing_set)

    if _classification:
        # Get accuracy and confusion matrix for training set
        all_actual_training = _get_classes(
            numpy.array([model_.activate(inp_vec) for inp_vec in training_set[0]]))
        all_expected_training = _get_classes(training_set[1])

        stats['training_accuracy'] = _get_accuracy(
            all_actual_training, all_expected_training)
        stats['training_confusion_matrix'] = _get_confusion_matrix(
            all_actual_training, all_expected_training, training_set[0].shape[1])

        # Get accuracy and confusion matrix for testing set
        all_actual_testing = _get_classes(
            numpy.array([model_.activate(inp_vec) for inp_vec in testing_set[0]]))
        all_expected_testing = _get_classes(testing_set[1])

        stats['testing_accuracy'] = _get_accuracy(
            all_actual_testing, all_expected_testing)
        stats['testing_confusion_matrix'] = _get_confusion_matrix(
            all_actual_testing, all_expected_testing, testing_set[0].shape[1])

    return stats


def _get_classes(all_outputs):
    """Return a list of classes given a matrix of outputs.

    We assume the class is the index of the highest output value for each output.
    """
    return numpy.argmax(all_outputs, axis=1)


def _get_accuracy(all_actual, all_expected):
    """Return the accuracy score for actual and expected classes.

    Args:
        all_actual: numpy.array<int>; An array of class indices.
        all_expected: numpy.array<int>; An array of class indices.

    Returns:
        float; Percent of matching classes.
    """
    # We want to count the number of matching classes,
    # and normalize by the number of classes
    return ((all_actual == all_expected).sum()
            / float(all_actual.size))


def _get_confusion_matrix(all_actual, all_expected, num_classes):
    """Return the confusion matrix for actual and expected classes.

    Args:
        all_actual: numpy.array<int>; An array of class indices.
        all_expected: numpy.array<int>; An array of class indices.

    Returns:
        numpy.array; Matrix with rows for expected classes, columns for actual
            classes, and cells with counts.
    """
    return numpy.bincount(num_classes * (all_expected) + (all_actual),
                          minlength=num_classes*num_classes).reshape(num_classes, num_classes)

############################
# Setup for Cross Validation
############################
def make_validation_sets(input_matrix, target_matrix, num_folds=3):
    return _create_train_test_sets(_split_dataset(input_matrix, target_matrix, num_folds))

def _split_dataset(input_matrix, target_matrix, num_sets):
    """Split patterns into num_sets disjoint sets."""
    # Zip for easier splitting
    patterns = zip(input_matrix, target_matrix)

    sets = []
    start_pos = 0
    set_size = len(patterns) / num_sets # rounded down
    for i in range(num_sets):
        # For the last set, add all remaining items (in case sets don't split evenly)
        if i == num_sets-1:
            new_set = patterns[start_pos:]
        else:
            new_set = patterns[start_pos:start_pos+set_size]

        # Transpose back into (input_matrix, target_matrix)
        set_input_matrix, set_target_matrix = zip(*new_set)
        sets.append((numpy.array(set_input_matrix), numpy.array(set_target_matrix)))

        start_pos += set_size

    return sets

def _create_train_test_sets(sets):
    """Organize sets into training and testing groups.

    Each group has one test set, and all others are training.
    Each group has all (input_matrix, target_matrix) between the train and test set.
    """
    train_test_sets = []
    num_folds = len(sets)

    for i in range(num_folds):
        test_set = sets[i]

        # Train set is all other sets
        train_set = [[], []]
        for j in range(num_folds):
            if i != j:
                train_set[0].extend(list(sets[j][0]))
                train_set[1].extend(list(sets[j][1]))

        # Train, test tuples
        train_set[0] = numpy.array(train_set[0])
        train_set[1] = numpy.array(train_set[1])
        train_test_sets.append((tuple(train_set), test_set))

    return train_test_sets

#############################
# Statistics
#############################
def _mean(list_):
    return sum(list_) / float(len(list_))

def _mean_of_dicts(dicts):
    """Obtain a mean dict from a list of dicts with the same keys.

    Args:
        dicts: list : dict; A list of dicts.
    """
    first = dicts[0]

    mean = {}
    for key in first:
        # Skip non numberic attributes
        if isinstance(first[key], (numbers.Number, numpy.ndarray)):
            mean[key] = _mean([dict_[key] for dict_ in dicts])

    return mean

def _sd(list_, mean):
    return numpy.sqrt(sum([(val - mean)**2 for val in list_]) / float(len(list_)))

def _sd_of_dicts(dicts, means):
    """Obtain a standard deviation dict from a list of dicts with the same keys.

    Note that this is the population standard deviation,
    not the sample standard deviation.

    Args:
        dicts: list : dict; A list of dicts.
        means: dict; dict mapping key to mean of key in dicts
    """
    first = dicts[0]

    standard_deviation = {}
    for key in first:
        # Skip non numberic attributes
        if isinstance(first[key], (numbers.Number, numpy.ndarray)):
            standard_deviation[key] = _sd([dict_[key] for dict_ in dicts], means[key])

    return standard_deviation


def _add_mean_sd_to_stats(stats, key='folds'):
    mean = _mean_of_dicts(stats[key])
    sd = _sd_of_dicts(stats[key], mean) 

    stats['mean'] = mean
    stats['sd'] = sd


def cross_validate(model_, dataset, num_folds=3, **kwargs):
    """Split the patterns, then train and test network on each fold."""

    # Get our sets, for use in cross validation
    train_test_sets = make_validation_sets(*dataset, num_folds=num_folds)

    # Get the stats on each set
    folds = []
    for i, (train_set, test_set) in enumerate(train_test_sets):
        if model_.logging:
            print 'Fold {}:'.format(i)

        folds.append(_validate_model(model_, train_set, test_set, **kwargs))

        if model_.logging:
            print

    stats = {'folds': folds}

    # Get average and standard deviation
    _add_mean_sd_to_stats(stats)

    return stats

def benchmark(model_, dataset, num_folds=3, num_runs=30, **kwargs):
    # TODO: maybe just take a function, and aggregate stats for that function

    runs = []
    for _ in range(num_runs):
        runs.append(cross_validate(model_, dataset, num_folds, **kwargs))
    stats = {'runs': runs}

    # Calculate meta stats
    means = [run['mean'] for run in runs]
    mean_of_means = _mean_of_dicts(means)
    sd_of_means = _sd_of_dicts(means, mean_of_means)
    stats['mean_of_means'] = mean_of_means
    stats['sd_of_means'] = sd_of_means

    return stats

def compare(names_models_datasets_kwargs, num_folds=3, num_runs=30):
    """Compare a set of algorithms on a set of patterns.

    Args:
        names_models_datasets_kwargs: list : tuple;
            list of (name, model, (input_matrix, target_matrix), kwargs) tuples.
        num_folds: int; number of folds for each cross validation test.
        num_runs: int; number of runs for each benchmark.
    """

    stats = {}
    for (name, model_, patterns, kwargs) in names_models_datasets_kwargs:
        stats[name] = benchmark(model_, patterns, num_folds, num_runs, **kwargs)

    # Calculate meta stats
    means = [results['mean_of_means'] for results in stats.itervalues()]

    # TODO: calculate means and sds from all folds, instead of means of folds
    mean_of_means = _mean_of_dicts(means)
    sd_of_means = _sd_of_dicts(means, mean_of_means)
    stats['mean_of_means'] = mean_of_means
    stats['sd_of_means'] = sd_of_means

    return stats
