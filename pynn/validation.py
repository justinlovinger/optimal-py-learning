import math
import time
import numbers

def _validate_network(network_, training_set, testing_set, **kwargs):
    """Test the given network on a partitular trainign and testing set."""

    # Temporarily disable logging
    curr_logging = network_.logging
    network_.logging = False

    # Train network on training set
    network_.reset()

    start = time.clock()
    network_.train(training_set, **kwargs) # Train
    elapsed = time.clock() - start

    network_.logging = curr_logging # Logging back to previous

    # Collect stats
    stats = {}
    stats['time'] = elapsed
    stats['epochs'] = network_.iteration

    # Get error for training and testing set
    stats['training_error'] = network_.get_avg_error(training_set)
    stats['testing_error'] = network_.get_avg_error(testing_set)

    return stats

############################
# Setup for Cross Validation
############################
def _split_dataset(patterns, num_sets):
    """Split patterns into num_sets disjoint sets."""

    sets = []
    start_pos = 0
    set_size = len(patterns)/num_sets # rounded down
    for i in range(num_sets):
        # For the last set, add all remaining items (in case sets don't split evenly)
        if i == num_sets-1:
            new_set = patterns[start_pos:]
        else:
            new_set = patterns[start_pos:start_pos+set_size]
        sets.append(new_set)
        start_pos += set_size
    return sets


def _create_train_test_sets(sets):
    """Organize sets into training and testing groups.

    Each group has one test set, and all others are training.
    Each group has all patterns between the train and test set.
    """

    train_test_sets = []
    num_folds = len(sets)

    for i in range(num_folds):
        test_set = sets[i]

        # Train set is all other sets
        train_set = []
        for j in range(num_folds):
            if i != j:
                train_set.extend(sets[j])

        # Train, test tuples
        train_test_sets.append((train_set, test_set))

    return train_test_sets

#############################
# Statistics
#############################
def _mean(list_):
    return sum(list_) / len(list_)

def _mean_of_dicts(dicts):
    """Obtain a mean dict from a list of dicts with the same keys.

    Args:
        dicts: list : dict; A list of dicts.
    """
    first = dicts[0]

    mean = {}
    for key in first:
        # Skip non numberic attributes
        if isinstance(first[key], numbers.Number):
            mean[key] = _mean([dict_[key] for dict_ in dicts])

    return mean

def _sd(list_, mean):
    return math.sqrt(sum([(val - mean)**2 for val in list_]) / len(list_))

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
        if isinstance(first[key], numbers.Number):
            standard_deviation[key] = _sd([dict_[key] for dict_ in dicts], means[key])

    return standard_deviation


def _add_mean_sd_to_stats(stats, key='folds'):
    mean = _mean_of_dicts(stats[key])
    sd = _sd_of_dicts(stats[key], mean) 

    stats['mean'] = mean
    stats['sd'] = sd


def cross_validate(network_, patterns, num_folds=3, **kwargs):
    """Split the patterns, then train and test network on each fold."""

    # Get our sets, for use in cross validation
    sets = _split_dataset(patterns, num_folds)
    train_test_sets = _create_train_test_sets(sets)

    # Get the stats on each set
    stats = {}

    folds = []
    for (train_set, test_set) in train_test_sets:
        folds.append(_validate_network(network_, train_set, test_set, **kwargs))
    stats['folds'] = folds

    # Get average and standard deviation
    _add_mean_sd_to_stats(stats)

    return stats

def benchmark(network_, patterns, num_folds=3, num_runs=30, **kwargs):
    # TODO: maybe just take a function, and aggregate stats for that function
    
    runs = []
    for i in range(num_runs):
        runs.append(cross_validate(network_, patterns, num_folds, **kwargs))
    stats = {'runs': runs}

    # Calculate meta stats
    means = [run['mean'] for run in runs]
    mean_of_means = _mean_of_dicts(means)
    sd_of_means = _sd_of_dicts(means, mean_of_means)
    stats['mean_of_means'] = mean_of_means
    stats['sd_of_means'] = sd_of_means

    return stats

def compare(name_networks_patterns_kwargs, num_folds=3, num_runs=30):
    """Compare a set of algorithms on a set of patterns.
    
    Args:
        name_networks_patterns_kwargs: list : tuple; list of (name, network, patterns, kwargs) tuples.
        num_folds: int; number of folds for each cross validation test.
        num_runs: int; number of runs for each benchmark.
    """

    stats = {}
    for (name, network_, patterns, kwargs) in name_networks_patterns_kwargs:
        stats[name] = benchmark(network_, patterns, num_folds, num_runs, **kwargs)

    # Calculate meta stats
    means = [results['mean_of_means'] for results in stats.itervalues()]
    mean_of_means = _mean_of_dicts(means)
    sd_of_means = _sd_of_dicts(means, mean_of_means)
    stats['mean_of_means'] = mean_of_means
    stats['sd_of_means'] = sd_of_means

    return stats