import copy

def normalize(input_vectors):
    """Normalize all inputs to a mean of 0 and equal variance, for each dimension.
    
    This improves numerical stability and allows for easier gradient descent.
    
    Args:
        input_vectors: list of input vectors.
    """
    raise NotImplementedError()

def clean_dataset_depuration(dataset, k=3, k_prime=2):
    """Clean a dataset with the Depuration procedure.

    See section 3.1 of "Analysis of new techniques to obtain quality training sets".
    """
    raise NotImplementedError()

    assert (k + 1) / 2 <= k_prime and k_prime <= k

    cleaned_dataset = []
    for point in dataset:
        pass
        # Find k-NN of point in dataset - {point]
        # if a class has at least k_prime representatives
        # among the k neighbours
            # Change the label of point to that class
            # Add point to cleaned_dataset
        # else
            # discard point (do not add to cleaned_dataset)