""" CS5340 Lab 4 Part 2: Gibbs Sampling
See accompanying PDF for instructions.

Name: Liu Yichao
Email: yichao.liu@u.nus.edu
Student ID: A0304386A
"""


import copy
import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentParser
from factor_utils import factor_evidence, factor_marginalize, assignment_to_index
from factor import Factor


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, 'ground-truth')

""" HELPER FUNCTIONS HERE """

def _get_markov_blanket(node, edges):
    """
    Returns the Markov blanket of a node in a DAG represented by edges, 
    including the node itself.

    Args:
        node: node of interest
        edges: numpy array of unduplicated edges, e.g. [i, j] implies that 
            node `i` is the parent of node `j`.

    Returns:
        numpy array of nodes in the Markov blanket of the node
    """
    markov_blanket = np.array([node], dtype=np.int64)
    parents = edges[np.where(edges[:, 1] == node)][:, 0]
    children = edges[np.where(edges[:, 0] == node)][:, 1]
    for child in children:
        siblings = edges[np.where(edges[:, 1] == child)][:, 0]
        markov_blanket = np.concatenate([markov_blanket, siblings])
    markov_blanket = np.concatenate([parents, children, markov_blanket])
    markov_blanket = np.unique(markov_blanket).astype(np.int64)
    return markov_blanket

""" END HELPER FUNCTIONS HERE"""


def _sample_step(nodes, factors, in_samples):
    """
    Performs gibbs sampling for a single iteration. Returns a sample for each node

    Args:
        nodes: numpy array of nodes
        factors: dictionary of factors e.g. factors[x1] returns the local factor for x1
        in_samples: dictionary of input samples (from previous iteration)

    Returns:
        dictionary of output samples where samples[x1] returns the sample for x1.
    """
    samples = copy.deepcopy(in_samples)

    """ YOUR CODE HERE """
    for node in nodes:
        samples.pop(node)
        factor = factor_evidence(factors[node], samples)
        node_sample = np.random.choice(
            factor.card[0], size=1, p=factor.val/np.sum(factor.val)
        )[0]
        samples[node] = node_sample

    """ END YOUR CODE HERE """

    return samples


def _get_conditional_probability(nodes, edges, factors, evidence, initial_samples, num_iterations, num_burn_in):
    """
    Returns the conditional probability p(Xf | Xe) where Xe is the set of observed nodes and Xf are the query nodes
    i.e. the unobserved nodes. The conditional probability is approximated using Gibbs sampling.

    Args:
        nodes: numpy array of nodes e.g. [x1, x2, ...].
        edges: numpy array of edges e.g. [i, j] implies that nodes[i] is the parent of nodes[j].
        factors: dictionary of Factors e.g. factors[x1] returns the conditional probability of x1 given all other nodes.
        evidence: dictionary of evidence e.g. evidence[x4] returns the provided evidence for x4.
        initial_samples: dictionary of initial samples to initialize Gibbs sampling.
        num_iterations: number of sampling iterations
        num_burn_in: number of burn-in iterations

    Returns:
        returns Factor of conditional probability.
    """
    assert num_iterations > num_burn_in
    conditional_prob = Factor()

    """ YOUR CODE HERE """
    # update factors with evidence 
    factors = {
        node: factor_evidence(factor, evidence)
        for node, factor in factors.items()
    }
    ne_nodes = np.array([node for node in nodes if node not in evidence])
    card = factors[nodes[0]].card

    # reduce factors within markov blanket
    for node in ne_nodes:
        mb_nodes = _get_markov_blanket(node, edges)
        not_mb_nodes = np.array([node for node in ne_nodes if node not in mb_nodes])
        factors[node] = factor_marginalize(factors[node], not_mb_nodes)

    # remove evidence nodes from initial samples
    initial_samples = {
        node: initial_samples[node]
        for node in ne_nodes
    }

    # print(ne_nodes)
    # for factor in factors.values():
    #     print(factor)
    # assert False

    # burn-in step
    for i in range(num_burn_in):
        initial_samples = _sample_step(ne_nodes, factors, initial_samples)
    
    # Gibbs sampling
    input_samples = initial_samples
    samples = np.array([]) # store samples as assignment_to_index array w.r.t. `nodes`
    # all factors have the same cardinality w.r.t. the ascending order of `ne_nodes`
    for i in tqdm(range(num_iterations)):
        input_samples = _sample_step(ne_nodes, factors, input_samples)
        # change `input_samples` from dictionary {node:sample} to array
        # [sample1, sample2, ...] w.r.t. the order of `ne_nodes`
        assignment = np.array([input_samples[node] for node in ne_nodes])
        samples = np.append(samples, assignment_to_index(assignment, card))
    
    # calculate frequency of each sample
    counter = Counter(samples)
    freq = np.array([counter[i] for i in range(np.prod(card))])
    freq = freq / np.sum(freq)
    conditional_prob.var = ne_nodes
    conditional_prob.card = card
    conditional_prob.val = freq

    """ END YOUR CODE HERE """

    return conditional_prob


def load_input_file(input_file: str) -> (Factor, dict, dict, int):
    """
    Returns the target factor, proposal factors for each node and evidence. DO NOT EDIT THIS FUNCTION

    Args:
        input_file: input file to open

    Returns:
        Factor of the target factor which is the target joint distribution of all nodes in the Bayesian network
        dictionary of node:Factor pair where Factor is the proposal distribution to sample node observations. Other
                    nodes in the Factor are parent nodes of the node
        dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
    """
    with open(input_file, 'r') as f:
        input_config = json.load(f)
    proposal_factors_dict = input_config['proposal-factors']

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    nodes = np.array(input_config['nodes'], dtype=int)
    edges = np.array(input_config['edges'], dtype=int)
    node_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                    node, proposal_factor_dict in proposal_factors_dict.items()}

    evidence = {int(node): ev for node, ev in input_config['evidence'].items()}
    initial_samples = {int(node): initial for node, initial in input_config['initial-samples'].items()}

    num_iterations = input_config['num-iterations']
    num_burn_in = input_config['num-burn-in']
    return nodes, edges, node_factors, evidence, initial_samples, num_iterations, num_burn_in


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()
    # np.random.seed(0)

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    nodes, edges, node_factors, evidence, initial_samples, num_iterations, num_burn_in = \
        load_input_file(input_file=input_file)

    # solution part
    from time import time
    start = time()
    conditional_probability = _get_conditional_probability(nodes=nodes, edges=edges, factors=node_factors,
                                                           evidence=evidence, initial_samples=initial_samples,
                                                           num_iterations=num_iterations, num_burn_in=num_burn_in)
    print(conditional_probability)
    print('Time: %.2fs'%(time() - start))
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    save__dict = {
        'var': np.array(conditional_probability.var).astype(int).tolist(),
        'card': np.array(conditional_probability.card).astype(int).tolist(),
        'val': np.array(conditional_probability.val).astype(float).tolist()
    }

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(save__dict, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
