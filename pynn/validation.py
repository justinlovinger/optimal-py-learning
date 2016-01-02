import math
import time

def validate_set(network, training_set, testing_set, **kwargs):
    stats = {}

    # Train network on training set
    start = time.clock()
    network.train(training_set, **kwargs)
    elapsed = time.clock() - start

    stats['time'] = elapsed
    stats['epochs'] = network.iteration

    # Get error for training and testing set
    stats['training_error'] = network.get_error(training_set)
    stats['testing_error'] = network.get_error(testing_set)

    return stats

def split_dataset(patterns, num_sets):
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

def create_train_test_sets(sets):
    """Organize sets into training and testing groups.

    Each group has a train and a test set.
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

def mean_of_folds(stats):
    num_folds = len(stats.keys())

    mean = {}
    for key in stats['Fold 0']:
        mean[key] = sum(fold[key] for fold in stats.values())/float(num_folds)
    return mean

def std_of_folds(stats, mean):
    num_folds = len(stats.keys())

    std = {}
    for key in stats['Fold 0']:
        std[key] = math.sqrt(sum((fold[key]-mean[key])**2 for fold in stats.values())/float(num_folds))
    return std

def add_mean_std_to_stats(stats):
    mean = mean_of_folds(stats)
    std = std_of_folds(stats, mean) 

    stats['Mean'] = mean
    stats['STD'] = std

def cross_validate(network, patterns, num_folds=3,
                   iterations=1000, error_break=0.02):
    """Split the patterns, then train and test network on each fold."""

    # Get our sets, for use in cross validation
    sets = split_dataset(patterns, num_folds)
    train_test_sets = create_train_test_sets(sets)

    # Get the stats on each set
    stats = {}

    for i, (train_set, test_set) in enumerate(train_test_sets):
        network.reset()
        stats['Fold {}'.format(i)] = validate_set(network, train_set, test_set,
                                                  iterations=iterations, error_break=error_break)

    # Get average and std
    add_mean_std_to_stats(stats)

    return stats