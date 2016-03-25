from pynn.data import datasets

def test_lenses():
    patterns = datasets.get_lenses()
    assert patterns[0] == ([-1, -1, -1, -1], [0, 0, 1])
    assert patterns[1] == ([-1, -1, -1, 1], [0, 1, 0])
    assert patterns[8] == ([0, -1, -1, -1], [0, 0, 1])