import copy

import numpy

from pynn.architecture import knn

def normalize(data_matrix):
    """Normalize matrix to a mean of 0 and standard devaiation of 1, for each dimension.
    
    This improves numerical stability and allows for easier gradient descent.
    
    Args:
        data_matrix: numpy.matrix; A matrix of values.
            We expect each row to be a point, and each column to be a dimension.
    """
    np_data_matrix = numpy.array(data_matrix)
    np_data_matrix -= numpy.mean(np_data_matrix, 0)
    np_data_matrix /= numpy.std(np_data_matrix, 0)
    return np_data_matrix

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
    changed_points = []
    removed_points = []
    for i, point in enumerate(dataset):
        # Find k-NN of point in dataset - {point}
        k_nearest = knn.select_k_nearest_neighbors(_list_minus_i(dataset, i),
                                                         point[0], k)
        
        # if a class has at least k_prime representatives
        # among the k neighbours
        class_counts = _count_classes(k_nearest)
        removed = True
        for class_, count in class_counts.iteritems():
            if count >= k_prime:
                # Change the label of point to that class
                # and add point to cleaned_dataset

                # Make new tuple from original inputs,
                # and the common class
                new_point = (point[0], class_)

                cleaned_dataset.append(new_point)

                # Check if new point is different than old point, track changed
                if class_ != tuple(point[1]):
                    changed_points.append(i)

                removed = False
                break
        if removed:
            # discard point (do not add to cleaned_dataset)
            removed_points.append(i)

    return cleaned_dataset, changed_points, removed_points

def pca_get_eigenvalues(data_matrix):
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

def pca_reduce_dimensions(data_matrix, eigen_values, eigen_vectors,
                          desired_num_dimensions):
    """Use principle component analysis to reduce the dimensionality of a data set.
    
    Note: dataset is normalized before analysis, without side effects.
    """
    # Key is an array of indexes, each index corresponds to an
    # eigenvalue and eigenvector
    key = numpy.argsort(eigen_values)[::-1][:desired_num_dimensions]

    # Numpy lets us use this key to easily construct new arrays with
    # only the chosen indexes. The [:, key] slice selects columns with the
    # given indexes
    eigen_vectors = eigen_vectors[:, key]

    # Perform the reduction using selected eigenvectors
    reduced_data_matrix = numpy.dot(eigen_vectors.T, data_matrix.T).T
    return reduced_data_matrix

def pca(data_matrix, desired_num_dimensions=None, num_dimensions_func=None):
    """Use principle component analysis to reduce the dimensionality of a data set.
    
    Note: dataset is normalized before analysis, without side effects,
          and resulting matrix is normalized before returning.
    """
    if desired_num_dimensions is not None and num_dimensions_func is not None:
        raise ValueError('Use only desired_num_dimensions or num_dimensions_func')
    if desired_num_dimensions is None and num_dimensions_func is None:
        raise ValueError('Use either desired_num_dimensions or num_dimensions_func')

    # Normalize
    normalized_matrix = normalize(data_matrix)

    # Perform PCA, using covariance method
    eigen_values, eigen_vectors = pca_get_eigenvalues(normalized_matrix)

    if num_dimensions_func is not None:
        desired_num_dimensions = num_dimensions_func(eigen_values)

    reduced_data_matrix = pca_reduce_dimensions(normalized_matrix, eigen_values, eigen_vectors,
                                                desired_num_dimensions)
    return normalize(reduced_data_matrix)
    

# Set default clean dataset function
def clean_dataset(dataset):
    cleaned_dataset, _, _ = clean_dataset_depuration(dataset)
    return cleaned_dataset
