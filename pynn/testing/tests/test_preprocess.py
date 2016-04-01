from pynn import preprocess
from pynn.data import datasets

def test_normalize():
    assert 0

def test_list_minus_i():
    list_ = [0, 1, 2]
    assert preprocess._list_minus_i(list_, 0) == [1, 2]
    assert preprocess._list_minus_i(list_, 1) == [0, 2]
    assert preprocess._list_minus_i(list_, 2) == [0, 1]

def test_count_classes():
    dataset = datasets.get_xor()
    class_counts = preprocess._count_classes(dataset)

    assert len(class_counts) == 2
    assert class_counts[0] == 2
    assert class_counts[1] == 2

    dataset = [
         ([0], 'foo'),
         ([0], 'bar'),
         ([0], 'bar')
        ]
    class_counts = preprocess._count_classes(dataset)

    assert len(class_counts) == 2
    assert class_counts['foo'] == 1
    assert class_counts['bar'] == 2

def test_clean_dataset_depuration():
    assert 0