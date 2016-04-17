from pynn import graph

def test_reverse_edge():
    edge = ('1', '2')
    assert graph._reverse_edge(edge) == ('2', '1')

    
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