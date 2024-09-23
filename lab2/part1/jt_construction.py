import numpy as np
import networkx as nx
from networkx.algorithms import tree
from factor import Factor
from factor_utils import factor_product


""" ADD HELPER FUNCTIONS HERE (IF NEEDED) """

def construct_graph(nodes, edges):
    '''
    Args:
        nodes: np array of nodes
        edges: np array of edges

    Returns:
        graph: dictionary of nodes and their neighbors(stored in sets)
    '''
    graph = {node: set() for node in nodes}
    for edge in edges:
        graph[edge[0]].add(edge[1])
        graph[edge[1]].add(edge[0])
    return graph


def triangulate_graph(graph):
    '''
    Args:
        graph: dictionary of nodes and their neighbors(stored in sets)

    Returns:
        cliques: a list of maximal cliques according to the elimination order
    '''
    cliques = []
    nodes = set(graph.keys())

    # aribitrarily choose an elimination order, here we just follow the order of nodes
    for node in list(nodes):
        neighbors = graph[node] & nodes
        # record the new maximal clique
        cliques.append(neighbors | {node})
        # connect all neighbors of the eliminated node 
        for i in neighbors & nodes:
            for j in neighbors & nodes:
                if i != j and j not in graph[i]:
                    graph[i].add(j)   
                    graph[j].add(i)
        nodes.remove(node)  

    return cliques


def get_all_intersections_between_cliques(cliques):
    '''
    Get all edges between cliques with weights of the number of common vars
    Args:
        cliques: a list of maximal cliques

    Returns:
        edges: 2d np array of edges which represent the intersection vars between cliques
            with weights(the number of common vars)
    '''
    n = len(cliques)
    edges = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            common_vars = set(cliques[i]) & set(cliques[j])
            if common_vars:
                edges[i][j] = len(common_vars)
                edges[j][i] = len(common_vars)
    return np.array(edges)


# prim algorithm to find the maximal spanning tree of a graph
def prim(edges):
    '''
    Prim algorithm to find the maximal spanning tree of a graph
    Args:
        edges: 2d np array of edges which represent the intersection vars between cliques
            with weights(the number of common vars)

    Returns:
        tree_edges: 2d np array of edges in the maximal spanning tree
    '''
    n = edges.shape[0]
    tree_edges = []
    visited = [False] * n
    visited[0] = True
    for _ in range(n - 1): # add n-1 edges to form a tree
        max_weight = 0
        max_edge = None
        for i in range(n):
            if visited[i]:
                for j in range(n):
                    if not visited[j] and edges[i][j] > max_weight:
                        max_weight = edges[i][j]
                        max_edge = [i, j]
        assert max_edge is not None, 'The graph is not connected'
        tree_edges.append(max_edge)
        visited[max_edge[1]] = True
    return np.array(tree_edges)

""" END ADD HELPER FUNCTIONS HERE """


def _get_clique_factors(jt_cliques, factors):
    """
    Assign node factors to cliques in the junction tree and derive the clique factors.

    Args:
        jt_cliques: list of junction tree maximal cliques 
            e.g. [[x1, x2, x3], [x2, x3], ... ]
        factors: list of factors from the original graph

    Returns:
        list of clique factors where the factor(jt_cliques[i]) = clique_factors[i]
    """
    clique_factors = [Factor() for _ in jt_cliques]

    """ YOUR CODE HERE """
    for factor in factors:
        for i, clique in enumerate(jt_cliques):
            if set(factor.var).issubset(set(clique)):
                clique_factors[i] = factor_product(clique_factors[i], factor)
                break # each factor can only be assigned to one clique
    """ END YOUR CODE HERE """

    assert len(clique_factors) == len(jt_cliques), \
        'there should be equal number of cliques and clique factors'
    return clique_factors


def _get_jt_clique_and_edges(nodes, edges):
    """
    Construct the structure of the junction tree and return the list of cliques (nodes) 
    in the junction tree and the list of edges between cliques in the junction tree. 
    [i, j] in jt_edges means that cliques[j] is a neighbor of cliques[i] and vice versa. 
    [j, i] should also be included in the numpy array of edges if [i, j] is present.
    You can use nx.Graph() and nx.find_cliques().

    Args:
        nodes: numpy array of nodes [x1, ..., xN]
        edges: numpy array of edges e.g. [x1, x2] implies that x1 and x2 are neighbors.

    Returns:
        list of junction tree cliques. each clique should be a maximal clique. e.g. [[X1, X2], ...] 
        numpy array of junction tree edges e.g. [[0,1], ...], 
        [i,j] means that cliques[i] and cliques[j] are neighbors.
    """
    # jt_edges = np.array(edges)  # dummy value

    """ YOUR CODE HERE """
    # find all maximal cliques is NP-hard, we use bron-kerbosch algorithm here
    # construct graph use a adjacency list `graph`: {node: set(neighbors)}
    graph = construct_graph(nodes, edges)
    # choose an elimination list to triangulate the graph and get all maximal cliques
    maximal_cliques = []
    maximal_cliques = triangulate_graph(graph)
    # get all edges between cliques with the weights as the number of common vars
    all_weighted_edges_between_cliques = get_all_intersections_between_cliques(maximal_cliques)
    # use maximal spanning tree algorithm to construct a junction tree
    # we use prim algorithm here since the graph is dense(fully connected)
    jt_edges = prim(all_weighted_edges_between_cliques)
    """ END YOUR CODE HERE """

    return maximal_cliques, jt_edges


def construct_junction_tree(nodes, edges, factors):
    """
    Constructs the junction tree and returns its the cliques, edges and clique factors 
    in the junction tree. DO NOT EDIT THIS FUNCTION.

    Args:
        nodes: numpy array of random variables e.g. [X1, X2, ..., Xv]
        edges: numpy array of edges e.g. [[X1,X2], [X2,X1], ...]
        factors: list of factors in the graph

    Returns:
        list of cliques e.g. [[X1, X2], ...]
        numpy array of edges e.g. [[0,1], ...], [i,j] means that cliques[i] and cliques[j] 
            are neighbors.
        list of clique factors where jt_cliques[i] has factor jt_factors[i] where i is 
            an index
    """
    jt_cliques, jt_edges = _get_jt_clique_and_edges(nodes=nodes, edges=edges)
    jt_factors = _get_clique_factors(jt_cliques=jt_cliques, factors=factors)
    return jt_cliques, jt_edges, jt_factors
