###############################################################################
# The MIT License (MIT)
#
# Copyright (c) 2017 Justin Lovinger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

from learning import datasets
from learning.architecture import som


def test_som_reduces_distances():
    # SOM functions correctly if is moves neurons towards inputs
    input_matrix, target_matrix = datasets.get_xor()

    # Small initial weight range chosen so network isn't "accidentally"
    # very close to inputs initially (which could cause test to fail)
    som_ = som.SOM(2, 4, initial_weights_range=0.25)

    # Convenience function
    def min_distances():
        all_closest = []
        for inp_vec in input_matrix:
            distances = som_.activate(inp_vec)
            all_closest.append(min(distances))
        return all_closest

    # Train SOM
    # Assert that distances have decreased
    all_closest = min_distances()
    som_.train(input_matrix, target_matrix, iterations=20)
    new_closest = min_distances()
    print all_closest
    print new_closest
    for old_c, new_c in zip(all_closest, new_closest):
        assert new_c < old_c
