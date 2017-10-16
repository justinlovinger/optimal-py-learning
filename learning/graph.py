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

import copy


def _validate_edge(edge):
    if not isinstance(edge, tuple) or len(edge) != 2:
        raise TypeError('edge must be a (from_node, to_node) tuple')


def _reverse_edge(edge):
    return (edge[1], edge[0])


def _extract_nodes(adjacency_dict):
    """Return the set of all ndoes in this graph."""
    nodes = set()
    for from_node, connected_nodes in adjacency_dict.iteritems():
        nodes.add(from_node)
        for node in connected_nodes:
            nodes.add(node)
    return nodes


def _extract_edges(adjacency_dict):
    """Return the set of all edges in this graph."""
    edges = set()
    for from_node, connected_nodes in adjacency_dict.iteritems():
        for node in connected_nodes:
            edges.add((from_node, node))
    return edges


def _make_adjacency_dict(nodes, edges):
    # Initialize to empty lists
    adjacency_dict = {}
    for node in nodes:
        adjacency_dict[node] = []

    # Add edges
    for edge in edges:
        adjacency_dict[edge[0]].append(edge[1])

    return adjacency_dict


def _shallow_copy(adjacency_dict):
    """Create a copy of every list object, while maintaining items in lists."""
    new_dict = {}
    for key, list_of_values in adjacency_dict.iteritems():
        # Create a copy of the list itself
        new_dict[key] = list_of_values[:]
    return new_dict


class Graph(object):
    def __init__(self, adjacency_dict):
        # Maps node to connected nodes
        if not isinstance(adjacency_dict, dict):
            raise TypeError(
                'adjacency_dict must be a dict mapping node -> list of connected nodes'
            )
        for value in adjacency_dict.itervalues():
            if not (isinstance(value, list) or isinstance(value, tuple)):
                raise TypeError(
                    'adjacency_dict must be a dict mapping node -> list of connected nodes'
                )

        # Origionally, we deep copied the adjacency dict, to avoid side effects
        # However, when passing an adjacency dict of objects,
        # it is often desirable for the object ids in the graph to match
        # the corresponding objects outside the graph.
        # Performing a copy of the list object, with the same elements,
        # is a compromise
        self.adjacency = _shallow_copy(adjacency_dict)

        self.nodes = _extract_nodes(self.adjacency)
        self.edges = _extract_edges(self.adjacency)

        self.backwards_adjacency = self._make_backwards_adjacency()

    def _make_backwards_adjacency(self):
        backwards_edges = [_reverse_edge(edge) for edge in self.edges]
        return _make_adjacency_dict(self.nodes, backwards_edges)

    def add_edge(self, edge):
        """Add edge to graph.

        Args:
            edge: (from_node, to_node) tuple.
        """
        _validate_edge(edge)
        if edge in self.edges:
            raise ValueError('edge already in graph.')

        # Add to adjacency
        try:
            connected_nodes = self.adjacency[edge[0]]
            connected_nodes.append(edge[1])
        # from_node in edge is not yet a from_node in graph
        except KeyError:
            self.adjacency[edge[0]] = [edge[1]]

        # Add to set of edges, and set of nodes
        self.nodes.add(edge[0])
        self.nodes.add(edge[1])
        self.edges.add(edge)

    def remove_edge(self, edge):
        """Remove edge to graph.

        Args:
            edge: (from_node, to_node) tuple.
        """
        _validate_edge(edge)
        if not edge in self.edges:
            raise ValueError('edge not in graph.')

        # Remove from adjacency
        connected_nodes = self.adjacency[edge[0]]
        connected_nodes.remove(edge[1])

        # Remove from set of edges
        # Nodes remain in set of nodes, even if they have no edges left
        self.edges.remove(edge)


def find_path(adjacency_dict, start, end, path=[]):
    """Search for a path from start to end.

    From: https://www.python.org/doc/essays/graphs/

    Returns:
        list / None; If path exists, return list of nodes in path order, otherwise return None.
    """
    path = path + [start]
    if start == end:
        return path
    if not adjacency_dict.has_key(start):
        return None
    for node in adjacency_dict[start]:
        if node not in path:
            newpath = find_path(adjacency_dict, node, end, path)
            if newpath: return newpath
    return None


def traverse_breadth_first(adjacency_dict, start, node_callback):
    """Perform a breath first search to all reachable nodes, calling callback on each.

    Each node is visited only once.
    """
    open_list = [start]
    expanded_nodes = set()

    # Expand nodes that haven't been expanded, until we run out
    while len(open_list) > 0:
        # Expand the next node
        next_node = open_list.pop(0)
        expanded_nodes.add(next_node)

        # Perform callback on newly expanded node
        node_callback(next_node)

        # Expand the open list
        try:
            connected_nodes = adjacency_dict[next_node]
        except:
            connected_nodes = []
        for node in connected_nodes:
            # Don't add duplicates
            if node not in expanded_nodes and node not in open_list:
                open_list.append(node)


def find_reachable_nodes(adjacency_dict, start):
    """Perform a breath first search to discover all reachable nodes.

    Returns:
        set; The set of reachable nodes, including start.
    """
    # Add each discovered node to set
    expanded_nodes = set()

    def node_callback(node):
        expanded_nodes.add(node)

    traverse_breadth_first(adjacency_dict, start, node_callback)
    return expanded_nodes
