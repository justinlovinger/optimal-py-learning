from pynn import graph

def test_reverse_edge():
    edge = ('1', '2')
    assert graph._reverse_edge(edge) == ('2', '1')

    
def test_shallow_copy():
    assert 0


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
    assert 0