import random

from pynn import validation

def random_dataset():
    dataset = []
    inputs = random.randint(2, 5)
    targets = random.randint(1, 3)
    # Dataset of size between 100 - 150
    for i in range(random.randint(100, 150)):
        input = []
        for j in range(inputs):
            input.append(random.uniform(-1.0, 1.0))
        target = []
        for j in range(targets):
            target.append(random.uniform(-1.0, 1.0))
        
        # Each datapoint is an input, target pair
        dataset.append([input, target])

    return dataset

def test_split_dataset():
    dataset = random_dataset()
    num_sets = random.randint(2, 5)
    sets = validation.split_dataset(dataset, num_sets)

    # The right number of sets is created
    assert len(sets) == num_sets

    for i in range(num_sets):
        for j in range(i+1, num_sets):
            # Check that each set is about equal in size
            assert len(sets[i]) >= len(sets[j])-5 and len(sets[i]) <= len(sets[j])+5

            # Check that each set has unique datapoints
            for point1 in sets[i]:
                for point2 in sets[j]:
                    assert point1 != point2