def test_bagger():
    # Create dummy layers that return set outputs
    outputs = [[0, 1, 2], [1, 2, 3]]
    # TODO create dummy layers from outputs, set layers to bagger


    # Assert bagger returns average of those outputs
    output = None
    assert output == [0.5, 1.5, 2.5]

    assert 0