""" CS5340 Lab 2 Part 1: Junction Tree Algorithm
See accompanying PDF for instructions.

Name: Liu Yichao
Email: yichao.liu@u.nus.edu 
Student ID: A0304386A
"""

import os
import numpy as np
import json
import networkx as nx
from argparse import ArgumentParser

from factor import Factor
from jt_construction import construct_junction_tree
from factor_utils import factor_product, factor_evidence, factor_marginalize

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')  # we will store the input data files here!
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')  # we will store the prediction files here!


""" ADD HELPER FUNCTIONS HERE """


""" END HELPER FUNCTIONS HERE """


def _update_mrf_w_evidence(all_nodes, evidence, edges, factors):
    """
    Update the MRF graph structure from observing the evidence

    Args:
        all_nodes: numpy array of nodes in the MRF
        evidence: dictionary of node:observation pairs where evidence[x1] returns 
            the observed value of x1
        edges: numpy array of edges in the MRF
        factors: list of Factors in teh MRF

    Returns:
        numpy array of query nodes
        numpy array of updated edges (after observing evidence)
        list of Factors (after observing evidence; empty factors should be removed)
    """

    query_nodes = all_nodes
    updated_edges = edges
    updated_factors = factors # it's all shallow copy here

    """ YOUR CODE HERE """
    # remove evidence nodes from all nodes to get query nodes
    query_nodes = np.array([node for node in all_nodes if node not in evidence.keys()])
    
    # add edges between the neighbors of evidence node to update the MRF graph
    for eliminated_node in evidence.keys():
        neighbors0 = np.array(
            [edge[1] for edge in updated_edges if edge[0] == eliminated_node]
        )
        neighbors1 = np.array(
            [edge[0] for edge in updated_edges if edge[1] == eliminated_node]
        )
        neighbors_of_eliminated_node = \
            np.unique(np.concatenate((neighbors0, neighbors1)))
        for i in range(neighbors_of_eliminated_node.size):
            for j in range(i+1, neighbors_of_eliminated_node.size):
                new_edge = [neighbors_of_eliminated_node[i], neighbors_of_eliminated_node[j]]
                updated_edges = np.append(updated_edges, np.array([new_edge]), axis=0)
    # remove edges related to evidence nodes
    updated_edges = [edge for edge in updated_edges \
        if edge[0] not in evidence.keys() and edge[1] not in evidence.keys()]
    updated_edges = np.array(updated_edges)

    # update factors with evidence
    for i in range(len(updated_factors)):
        updated_factors[i] = factor_evidence(updated_factors[i], evidence)
    # remove empty factors after observing evidence, this step is not necessary
    updated_factors = [factor for factor in updated_factors if factor.var.size > 0]

    """ END YOUR CODE HERE """

    return query_nodes, updated_edges, updated_factors


def _get_clique_potentials(jt_cliques, jt_edges, jt_clique_factors):
    """
    Returns the list of clique potentials after performing the sum-product algorithm 
    on the junction tree

    Args:
        jt_cliques: list of junction tree nodes e.g. [[x1, x2], ...]
        jt_edges: numpy array of junction tree edges e.g. [i,j] implies that 
            jt_cliques[i] and jt_cliques[j] are neighbors
        jt_clique_factors: list of clique factors where jt_clique_factors[i] is 
            the factor for cliques[i]

    Returns:
        list of clique potentials computed from the sum-product algorithm
    """
    clique_potentials = jt_clique_factors

    """ YOUR CODE HERE """
    # construct clique graph
    n = len(jt_cliques)
    clique_graph = nx.Graph()
    clique_graph.add_nodes_from(range(n))
    clique_graph.add_edges_from(jt_edges)

    # intialize messages
    messages = [[Factor() for _ in range(n)] for _ in range(n)]

    def collect(i, j):
        '''
        i is the parent of j, collect messages from j to i and stored in messages[j][i]
        '''
        if(clique_graph.degree(j) == 1):
            messages[j][i] = factor_marginalize( 
                jt_clique_factors[j], 
                np.setdiff1d(list(jt_cliques[j]), list(jt_cliques[i]))
            )
        else:
            messages[j][i] = jt_clique_factors[j]
            for neighbor in clique_graph.neighbors(j):
                if neighbor != i:
                    collect(j, neighbor)
                    messages[j][i] = factor_product(messages[j][i], messages[neighbor][j])
            messages[j][i] = factor_marginalize(
                messages[j][i],
                np.setdiff1d(list(jt_cliques[j]), list(jt_cliques[i]))
            )

    def distribute(i, j):
        '''
        i is the parent of j, distribute messages from i to j and stored in messages[i][j]
        '''
        messages[i][j] = jt_clique_factors[i]
        for neighbor in clique_graph.neighbors(i):
            if neighbor != j:
                messages[i][j] = factor_product(messages[i][j], messages[neighbor][i])
        messages[i][j] = factor_marginalize(
            messages[i][j],
            np.setdiff1d(list(jt_cliques[i]), list(jt_cliques[j]))
        )
        for neighbor in clique_graph.neighbors(j):
            if neighbor != i:
                distribute(j, neighbor)
    
    root = 0
    # collect messages
    for neighbor in clique_graph.neighbors(root):
        collect(root, neighbor)
    # distribute messages
    for neighbor in clique_graph.neighbors(root):
        distribute(root, neighbor)

    # compute clique potentials
    for i in range(n):
        # this line is already done in the given beginning code
        # clique_potentials[i] = jt_clique_factors[i] 
        for neighbor in clique_graph.neighbors(i):
            clique_potentials[i] = factor_product(clique_potentials[i], messages[neighbor][i])

    """ END YOUR CODE HERE """

    assert len(clique_potentials) == len(jt_cliques)
    return clique_potentials


def _get_node_marginal_probabilities(query_nodes, cliques, clique_potentials):
    """
    Returns the marginal probability for each query node from the clique potentials.

    Args:
        query_nodes: numpy array of query nodes e.g. [x1, x2, ..., xN]
        cliques: list of cliques e.g. [[x1, x2], ... [x2, x3, .., xN]]
        clique_potentials: list of clique potentials (Factor class)

    Returns:
        list of node marginal probabilities (Factor class)

    """
    query_marginal_probabilities = [None for _ in query_nodes]

    """ YOUR CODE HERE """
    # brute-force marginalization and normalization 
    # for X_q, find an arbitrary clique that contains X_q, and marginalize out all 
    # other variables, then normalize. Since clique potentials are equivalent to 
    # joint probabilities, any clique that contains X_q will give the correct answer.

    # a more efficient way is to sort the cliques by size and then do inference in 
    # the order of clique size, thus we can avoid redundant computation
    is_marginalized = {X_q: False for X_q in query_nodes}
    sorted_cliques_with_potentials = \
        sorted(zip(cliques, clique_potentials), key=lambda x: len(x[0]))
    sorted_cliques, sorted_clique_potentials = zip(*sorted_cliques_with_potentials)
    for i, clique in enumerate(sorted_cliques):
        for X_q in clique: # every node inside a clique is a query node
            if not is_marginalized[X_q]:
                query_marginal = factor_marginalize(
                    sorted_clique_potentials[i], 
                    np.setdiff1d(list(clique), [X_q])
                )
                query_marginal.val = query_marginal.val.astype(np.float64)
                query_marginal.val /= np.sum(query_marginal.val)
                query_marginal_probabilities[np.where(X_q == query_nodes)[0][0]] = query_marginal
                is_marginalized[X_q] = True
        if all(is_marginalized.values()):
            break

    """ END YOUR CODE HERE """

    return query_marginal_probabilities


def get_conditional_probabilities(all_nodes, evidence, edges, factors):
    """
    Returns query nodes and query Factors representing the conditional probability of 
    each query node given the evidence e.g. p(xf|Xe) where xf is a single query node 
    and Xe is the set of evidence nodes.

    Args:
        all_nodes: numpy array of all nodes (random variables) in the graph
        evidence: dictionary of node:evidence pairs e.g. evidence[x1] returns the observed
            value for x1
        edges: numpy array of all edges in the graph e.g. [[x1, x2],...] implies that x1 
            is a neighbor of x2
        factors: list of factors in the MRF.

    Returns:
        numpy array of query nodes
        list of Factor
    """
    query_nodes, updated_edges, updated_node_factors = _update_mrf_w_evidence( \
        all_nodes=all_nodes, evidence=evidence, edges=edges, factors=factors)

    jt_cliques, jt_edges, jt_factors = construct_junction_tree( \
        nodes=query_nodes, edges=updated_edges, factors=updated_node_factors)

    clique_potentials = _get_clique_potentials(\
        jt_cliques=jt_cliques, jt_edges=jt_edges, jt_clique_factors=jt_factors)

    query_node_marginals = _get_node_marginal_probabilities(\
        query_nodes=query_nodes, cliques=jt_cliques, clique_potentials=clique_potentials)

    return query_nodes, query_node_marginals


def parse_input_file(input_file: str):
    """ Reads the input file and parses it. DO NOT EDIT THIS FUNCTION. """
    with open(input_file, 'r') as f:
        input_config = json.load(f)

    nodes = np.array(input_config['nodes'])
    edges = np.array(input_config['edges'])

    # parse evidence
    raw_evidence = input_config['evidence']
    evidence = {}
    for k, v in raw_evidence.items():
        evidence[int(k)] = v

    # parse factors
    raw_factors = input_config['factors']
    factors = []
    for raw_factor in raw_factors:
        factor = Factor(var=np.array(raw_factor['var']), card=np.array(raw_factor['card']),
                        val=np.array(raw_factor['val']))
        factors.append(factor)
    return nodes, edges, evidence, factors


def main():
    """ Entry function to handle loading inputs and saving outputs. DO NOT EDIT THIS FUNCTION. """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    nodes, edges, evidence, factors = parse_input_file(input_file=input_file)

    # solution part:
    query_nodes, query_conditional_probabilities = get_conditional_probabilities(\
        all_nodes=nodes, edges=edges, factors=factors, evidence=evidence)

    predictions = {}
    for i, node in enumerate(query_nodes):
        probability = query_conditional_probabilities[i].val
        predictions[int(node)] = list(np.array(probability, dtype=float))

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))
    with open(prediction_file, 'w') as f:
        json.dump(predictions, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
