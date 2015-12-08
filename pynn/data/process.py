import re

def normalize_value(value, min, max):
    """Scale value from min to max, to -1.0 to 1.0."""
    return ((value-min)/(max-min))*2-1
        
def unnormalize_value(value, min, max):
    """Return normalized value to original value."""
    return ((value+1)/2.0)*(max-min)+min

def normalize_all_values(all_values):
    """Normalize all values in a 2d matrix.

    all_values = [
                  [0, 1, 2],
                  [3, 4, 5],
                  etc.
                 ]
    """
    mins = [min(col) for col in zip(*all_values)]
    maxes = [max(col) for col in zip(*all_values)]

    for values in all_values:
        for i in range(len(values)):
            values[i] = normalize_value(values[i], mins[i], maxes[i])

def normalize_inputs(patterns):
    """Normalize all inputs in patterns.

    patterns should be list of (inputs, outputs) tuples.
    """
    attributes = [pattern[0] for pattern in patterns]
    normalize_all_values(attributes)

def normalize_targets(patterns):
    """Normalize all targets in patterns.

    patterns should be list of (inputs, outputs) tuples.
    """
    targets = [pattern[1] for pattern in patterns]
    normalize_all_values(targets)

def get_attributes(line):
    line_processed = re.sub(r' +', ',', line.strip())
    return line_processed.split(',')

def get_data(file_name, attr_start_pos, attr_end_pos=-1, target_pos=-1, classification=True):
    if classification:
        # Get data from file
        data_file = open(file_name)
        # Determine the classes
        classes = set()
        for line in data_file:
            attributes = get_attributes(line)
            class_ = attributes[target_pos].strip()
            classes.add(class_)
        class_dict = {}
        for i, class_ in enumerate(sorted(classes)): # Sorted for easier validation
            class_dict[class_] = i

    # Obtain a data point from each line of the file
    patterns = []
    data_file = open(file_name)
    for line in data_file:
        attributes = get_attributes(line)

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

        patterns.append((input, output))

    #scale inputs data from -1.0 to 1.0
    normalize_inputs(patterns)

    return patterns