import copy

from pynn.architecture import knn

def normalize(input_vectors):
    """Normalize all inputs to a mean of 0 and equal variance, for each dimension.
    
    This improves numerical stability and allows for easier gradient descent.
    
    Args:
        input_vectors: list of input vectors.
    """
    assert 0

def softmax_normalize(input_vectors):
    """Normalize inputs, while reducing the influence of outliers.
    
    See https://en.wikipedia.org/wiki/Softmax_function
    """
    assert 0

def _list_minus_i(list_, i):
    """Return list without item i."""
    return list_[:i] + list_[i+1:]

def _count_classes(points):
    """Count how many times each class appears in a set of points."""
    # TODO: Fix to work with list of target values.
    # Should we simply count a set of target values as one target?
    # Or track a separate count for each position (targets[0], targets[1], etc.)?
    class_counts = {}
    for point in points:
        target = tuple(point[1])
        try:
            class_counts[target] += 1
        except KeyError:
            class_counts[target] = 1

    return class_counts

def clean_dataset_depuration(dataset, k=3, k_prime=2):
    """Clean a dataset with the Depuration procedure.

    See section 3.1 of "Analysis of new techniques to obtain quality training sets".
    """
    if not ((k + 1) / 2 <= k_prime and k_prime <= k):
        raise ValueError('k_prime must be between (k + 1) / 2 and k')

    cleaned_dataset = []
    for i, point in enumerate(dataset):
        # Find k-NN of point in dataset - {point}
        k_nearest = knn.select_k_nearest_neighbors(_list_minus_i(dataset, i),
                                                         point[0], k)
        
        # if a class has at least k_prime representatives
        # among the k neighbours
        class_counts = _count_classes(k_nearest)
        for class_, count in class_counts.iteritems():
            if count >= k_prime:
                # Change the label of point to that class
                # and add point to cleaned_dataset

                # Make new tuple from original inputs,
                # and the common class
                new_point = (point[0], class_)

                cleaned_dataset.append(new_point)
                break
        # else
            # discard point (do not add to cleaned_dataset)

    return cleaned_dataset

# Set default clean dataset function
clean_dataset = clean_dataset_depuration