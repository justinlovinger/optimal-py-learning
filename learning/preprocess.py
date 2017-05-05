import copy
import random

import numpy

from learning.architecture import knn

def shuffle(dataset):
    """Return shuffled (input_matrix, target_matrix) dataset.

    Only supports numpy matrices.
    For python matrices: zip, shuffle, unzip.
    """
    # Shuffle indices, and use shuffled indices to slice both matrices
    indices = range(len(dataset[0]))
    random.shuffle(indices)
    return dataset[0][indices], dataset[1][indices]

def make_onehot(vector):
    """Return a matrix of one-hot vectors from a vector of values.

    Use to convet a vector of class labels into a target_matrix.
    Each one-hot vector has a single 1.0, and many 0.0s.
    """
    # Find classes, and corresponding onehot vectors
    class_onehots = _class_indices_to_onehots(_class_indices(vector))

    # Add onehot vector to matrix for each value (class) in vector
    matrix = []
    for val in vector:
        # Append a copy of the onehot vector corresponding to this class
        matrix.append(class_onehots[str(val)].copy())

    return numpy.array(matrix)

def _class_indices(vector):
    """Return dict mapping class -> index.

    Classes get index in order of first appearance.
    """
    # Find all classes
    # Each unique row is considered a class, whether matrix is 1d, 2d, etc.
    # This alllows it to work with col vectors, or 1d vectors
    class_indices = {}
    index = 0
    for val in vector:
        key = str(val)
        if key not in class_indices:
            class_indices[key] = index
            index += 1
    return class_indices

def _class_indices_to_onehots(class_indices):
    """Return dict mapping class -> onehot_vec."""
    num_classes = len(class_indices)
    class_onehots = {}
    for class_, index in class_indices.iteritems():
        onehot = numpy.zeros(num_classes)
        onehot[index] = 1.0
        class_onehots[class_] = onehot
    return class_onehots

########################
# Normalization
########################
def rescale(matrix):
    """Scale each column to [-1, 1]."""
    scaled_matrix = numpy.array(matrix, dtype='float64')

    scaled_matrix -= numpy.min(scaled_matrix, axis=0) # Each col, min of 0
    scaled_matrix /= numpy.max(scaled_matrix, axis=0) # Each col, max of 1

    # Scale from [0, 1] to [-1, 1]
    scaled_matrix *= 2.0
    scaled_matrix -= 1.0

    return scaled_matrix

def normalize(matrix):
    """Normalize matrix to a mean of 0 and standard devaiation of 1, for each dimension.

    This improves numerical stability and allows for easier gradient descent.

    Args:
        matrix: numpy.matrix; A matrix of values.
            We expect each row to be a point, and each column to be a dimension.
    """
    np_matrix = numpy.array(matrix, dtype='float64')

    if np_matrix.shape[0] < 2:
        raise ValueError('Cannot normalize a matrix with only one row')

    # Subtract the mean of each attribute from that attribute, for each point
    np_matrix -= numpy.mean(np_matrix, axis=0)

    # Divide the standard deviation of each attribute from that attribute, for each point
    try:
        with numpy.errstate(divide='raise', invalid='raise'):
            np_matrix /= numpy.std(np_matrix, axis=0)
    except FloatingPointError:
        # STD of zero
        np_matrix /= numpy.std(np_matrix, axis=0)

        # Replace Nan (0 / inf) and inf (x / inf) with 0.0
        np_matrix[~ numpy.isfinite(np_matrix)] = 0.0

    return np_matrix

def softmax_normalize(matrix):
    """Normalize inputs, while reducing the influence of outliers.

    See https://en.wikipedia.org/wiki/Softmax_function
    """
    assert 0

###########################
# Depuration
###########################
def clean_dataset_depuration(input_matrix, target_matrix, k=3, k_prime=2):
    """Clean a dataset with the Depuration procedure.

    See section 3.1 of "Analysis of new techniques to obtain quality training sets".
    """
    if not isinstance(input_matrix, numpy.ndarray):
        input_matrix = numpy.array(input_matrix)
    if not isinstance(target_matrix, numpy.ndarray):
        target_matrix = numpy.array(target_matrix)

    patterns = zip(input_matrix, target_matrix)
    indices = range(len(input_matrix))

    if not ((k + 1) / 2 <= k_prime and k_prime <= k):
        raise ValueError('k_prime must be between (k + 1) / 2 and k')

    kept_inputs = []
    kept_targets = []
    changed_patterns = []
    removed_patterns = []
    for i in indices:
        # Find k-NN of patternIi in patterns - {pattern_i}
        # We do this by finding k+1 nearest indices, and ignoring index i
        k_nearest = knn.select_k_nearest_neighbors(input_matrix, input_matrix[i], k+1)
        k_nearest.remove(i)

        # if a class has at least k_prime representatives
        # among the k neighbours
        class_counts = _count_classes(target_matrix[k_nearest])
        removed = True
        for class_, count in class_counts.iteritems():
            if count >= k_prime:
                # Change the label of pattern to that class
                # and add pattern to cleaned_patterns

                # Keep input
                # and the common class
                kept_inputs.append(input_matrix[i])
                kept_targets.append(class_)

                # Check if new pattern is different than old pattern, track changed
                if class_ != tuple(target_matrix[i]):
                    changed_patterns.append(i)

                removed = False
                break
        if removed:
            # discard pattern (do not add to kept lists)
            removed_patterns.append(i)

    return ((numpy.array(kept_inputs), numpy.array(kept_targets)),
            changed_patterns, removed_patterns)

def _list_minus_i(list_, i):
    """Return list without item i."""
    return list_[:i] + list_[i+1:]

def _count_classes(target_matrix):
    """Count how many times each class appears in a set of points."""
    # TODO: Fix to work with list of target values.
    # Should we simply count a set of target values as one target?
    # Or track a separate count for each position (targets[0], targets[1], etc.)?
    class_counts = {}
    for target in target_matrix:
        key = tuple(target)
        try:
            class_counts[key] += 1
        except KeyError:
            class_counts[key] = 1

    return class_counts

#########################
# PCA
#########################
def _pca_get_eigenvalues(data_matrix):
    """Get the eigenvalues and eigenvectors of the covariance matrix.

    Args:
        data_matrix; An n by m matrix.
            We expect each row to be a data point, and each column to be a dimension

    Returns:
        numpy.array; Eigenvalues.
        numpy.array; Eigenvectors.
    """
    covariance = numpy.cov(data_matrix, rowvar=False)
    return numpy.linalg.eigh(covariance)

def _pca_reduce_dimensions(data_matrix, eigen_vectors,
                           selected_dimensions):
    """Use principle component analysis to reduce the dimensionality of a data set.

    Args:
        data_matrix: numpy.array; The n x m matrix to reduce.
        eigen_vectors: numpy.array; The array of eigenvectors from the
            covariance matrix of data_matrix.
        selected_dimensions: list; list of indexes, each corresponding to an
            eigenvector.
    """
    if not isinstance(selected_dimensions, (list, numpy.ndarray)):
        raise TypeError('selected_dimensions must be a list of indexes') 

    # Key is an array of indexes, each index corresponds to an
    # eigenvalue and eigenvector

    # Numpy lets us use this key to easily construct new arrays with
    # only the chosen indexes. The [:, key] slice selects columns with the
    # given indexes
    eigen_vectors = eigen_vectors[:, selected_dimensions]

    # Perform the reduction using selected eigenvectors
    return numpy.dot(eigen_vectors.T, data_matrix.T).T

def pca(data_matrix, desired_num_dimensions=None, select_dimensions_func=None):
    """Use principle component analysis to reduce the dimensionality of a data set.

    Note: dataset is normalized before analysis, without side effects,
          and resulting matrix is normalized before returning.
    """
    if desired_num_dimensions is not None and select_dimensions_func is not None:
        raise ValueError('Use only desired_num_dimensions or num_dimensions_func')
    if desired_num_dimensions is None and select_dimensions_func is None:
        raise ValueError('Use either desired_num_dimensions or num_dimensions_func')

    # Normalize
    normalized_matrix = normalize(data_matrix)

    # Perform PCA, using covariance method
    eigen_values, eigen_vectors = _pca_get_eigenvalues(normalized_matrix)

    # Select eigenvectors, based on user input
    if desired_num_dimensions is not None:
        selected_dimensions = numpy.argsort(eigen_values)[::-1][:desired_num_dimensions]

    if select_dimensions_func is not None:
        selected_dimensions = select_dimensions_func(eigen_values)

    # Perform the pca reduction
    reduced_data_matrix = _pca_reduce_dimensions(normalized_matrix, eigen_vectors,
                                                 selected_dimensions)
    return normalize(reduced_data_matrix)


##############################
# All in one dataset cleaning
##############################
def _pca_select_greater_than_one(eigen_values):
    # TODO: Base on quartile or some kind of average, not 1.0
    return [i for i, v in enumerate(eigen_values) if v > 1]

def clean_dataset(input_matrix, target_matrix):
    if not isinstance(input_matrix, numpy.ndarray):
        input_matrix = numpy.array(input_matrix)
    if not isinstance(target_matrix, numpy.ndarray):
        target_matrix = numpy.array(target_matrix)

    # Clean inputs
    if input_matrix.shape[1] > 1: # More than 1 input dimension
        # Reduce input dimensions
        # And normalize (normalization performed by pca)
        reduced_inputs = pca(input_matrix, select_dimensions_func=_pca_select_greater_than_one)
    else:
        # Just normalize
        reduced_inputs = normalize(input_matrix)

    # Clean erronous targets
    cleaned_dataset, _, _ = clean_dataset_depuration(reduced_inputs, target_matrix)
    return cleaned_dataset
