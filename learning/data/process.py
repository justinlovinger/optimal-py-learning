import re

import numpy

from learning import preprocess

def get_data(file_name, attr_start_pos, attr_end_pos=-1, target_pos=-1, classification=True):
    if classification:
        # Get data from file
        data_file = open(file_name)
        # Determine the classes
        classes = set()
        for line in data_file:
            attributes = _get_attributes(line)
            class_ = attributes[target_pos].strip()
            classes.add(class_)
        class_dict = {}
        for i, class_ in enumerate(sorted(classes)): # Sorted for easier validation
            class_dict[class_] = i

    # Obtain a data point from each line of the file
    input_matrix = []
    target_matrix = []

    data_file = open(file_name)
    for line in data_file:
        attributes = _get_attributes(line)

        try:
            input = [float(value) for value in attributes[attr_start_pos:attr_end_pos]]
        except ValueError:
            continue

        if classification:
            class_ = attributes[target_pos].strip()
            output = [0.0]*len(classes)
            # Class dict maps the name to the position
            # This position is given a value of 1.0
            output[class_dict[class_]] = 1.0
        else:
            output = [float(attributes[target_pos].strip())]

        input_matrix.append(input)
        target_matrix.append(output)

    # Make numpy arrays
    input_matrix = numpy.array(input_matrix)
    target_matrix = numpy.array(target_matrix)

    # Re-scale input matrix to [-1, 1]
    input_matrix = preprocess.rescale(input_matrix)
    # Same for target if it is regression
    if classification is False:
        target_matrix = preprocess.rescale(target_matrix)

    return input_matrix, target_matrix

def _get_attributes(line):
    line_processed = re.sub(r' +', ',', line.strip())
    return line_processed.split(',')
