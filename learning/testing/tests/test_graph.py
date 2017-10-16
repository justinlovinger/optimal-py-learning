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

from learning import graph


def test_reverse_edge():
    edge = ('1', '2')
    assert graph._reverse_edge(edge) == ('2', '1')


def test_shallow_copy():
    object_1 = object()
    object_2 = object()
    list_ = [object_2]
    adjacency_dict = {object_1: list_}

    graph_ = graph.Graph(adjacency_dict)
    assert object_1 in graph_.adjacency.keys()
    assert object_2 in graph_.adjacency[object_1]
    assert list_ is not graph_.adjacency[object_1] # List is different instance


def test_traverse_bredth_first_visit_each_node_once():
    adjacency_dict = {'1': ['2', '3', '4'],
                      '2': ['4'],
                      '3': ['4'],
                      '4': ['1']}

    visited = set()
    def node_callback(node):
        assert node not in visited
        visited.add(node)
    graph.traverse_bredth_first(adjacency_dict, '1', node_callback)


def test_find_reachable_nodes():
    adjacency_dict = {'1': ['2', '3'],
                      '2': ['4'],
                      '4': ['1'],
                      'a': ['1'],
                      'b': ['c']}
    assert graph.find_reachable_nodes(adjacency_dict, '1') == set(['1', '2', '3', '4'])


def test_remove_edge():
    adjacency_dict = {'1': ['2', '3']}
    graph_ = graph.Graph(adjacency_dict)

    graph_.remove_edge(('1', '2'))
    assert graph_.adjacency == {'1': ['3']}
    assert graph_.edges == set([('1', '3')])


def test_add_edge_from_node_in_graph():
    adjacency_dict = {'1': ['2']}
    graph_ = graph.Graph(adjacency_dict)

    graph_.add_edge(('1', '3'))
    assert graph_.adjacency == {'1': ['2', '3']}
    assert graph_.edges == set([('1', '2'), ('1', '3')])
    assert graph_.nodes == set(['1', '2', '3'])


def test_add_edge_from_node_not_in_graph():
    adjacency_dict = {'1': ['2']}
    graph_ = graph.Graph(adjacency_dict)

    graph_.add_edge(('2', '3'))
    assert graph_.adjacency == {'1': ['2'], '2': ['3']}
    assert graph_.edges == set([('1', '2'), ('2', '3')])
    assert graph_.nodes == set(['1', '2', '3'])


def test_backwards_adjacency():
    adjacency_dict = {'1': ['2', '3'],
                      '2': ['3']}
    graph_ = graph.Graph(adjacency_dict)

    assert set(graph_.backwards_adjacency.keys()) == set(['1', '2', '3'])
    assert graph_.backwards_adjacency['1'] == []
    assert graph_.backwards_adjacency['2'] == ['1']
    assert set(graph_.backwards_adjacency['3']) == set(['1', '2'])
