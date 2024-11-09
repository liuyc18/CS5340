""" CS5340 Lab 4 Part 1: Importance Sampling
See accompanying PDF for instructions.

Name: Liu Yichao
Email: yichao.liu@u.nus.edu
Student ID: A0304386A
"""

import os
import json
import numpy as np
import networkx as nx
from factor_utils import factor_evidence, factor_product, assignment_to_index
from factor import Factor
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """



""" END HELPER FUNCTIONS HERE """


def _sample_step(nodes, proposal_factors):
    """
    Performs one iteration of importance sampling where it should sample a sample for each node. The sampling should
    be done in topological order.

    Args:
        nodes: numpy array of nodes. nodes are sampled in the order specified in nodes
        proposal_factors: dictionary of proposal factors where proposal_factors[1] returns the
                sample distribution for node 1

    Returns:
        dictionary of node samples where samples[1] return the scalar sample for node 1.
    """
    samples = {}

    """ YOUR CODE HERE: Use np.random.choice """
    evidence = {}
    for node in nodes:
        # in topo order, all parents have been sampled (as evidence) before this node 
        factor = factor_evidence(proposal_factors[node], evidence)
        # only 1 var remains in factor after factor_evidence in topo order
        node_sample = np.random.choice(factor.card[0], size=1, p=factor.val/np.sum(factor.val))[0] 
        samples[node] = node_sample
        evidence[node] = node_sample
    """ END YOUR CODE HERE """

    assert len(samples.keys()) == len(nodes)
    return samples


def _get_conditional_probability(target_factors, proposal_factors, evidence, num_iterations):
    """
    Performs multiple iterations of importance sampling and returns the conditional distribution p(Xf | Xe) where
    Xe are the evidence nodes and Xf are the query nodes (unobserved).

    Args:
        target_factors: dictionary of node:Factor pair where Factor is the target distribution of the node.
                        Other nodes in the Factor are parent nodes of the node. The product of the target
                        distribution gives our joint target distribution.
        proposal_factors: dictionary of node:Factor pair where Factor is the proposal distribution to sample node
                        observations. Other nodes in the Factor are parent nodes of the node
        evidence: dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
        num_iterations: number of importance sampling iterations

    Returns:
        Approximate conditional distribution of p(Xf | Xe) where Xf is the set of query nodes (not observed) and
        Xe is the set of evidence nodes. Return result as a Factor
    """
    out = Factor()

    """ YOUR CODE HERE """
    graph = nx.DiGraph()
    # init graph structure with evidence
    for node, factor in target_factors.items(): 
        graph.add_node(node)
        if node not in evidence:
            for var in factor.var:
                if var != node:
                    graph.add_edge(var, node)

    # import matplotlib.pyplot as plt
    # nx.draw(graph, with_labels=True)
    # plt.show()
    # assert False, "stop here"

    # update proposal factors with evidence
    proposal_factors = {
        node: factor_evidence(factor, evidence) \
        for node, factor in proposal_factors.items() \
        if node not in evidence # we don't need q(x_u) where x_u is evidence
    }

    # get topological order of nodes. O(n + m)
    topo_nodes = np.array(list(nx.topological_sort(graph))) 
    nodes = np.array(list(target_factors.keys()))
    query_nodes = np.array([node for node in topo_nodes if node not in evidence])

    # merge basic configs of target factors to 
    # get basic config (var, card, val.shape) of `out` joint distribution factor
    # and sort as numerical ascending order
    # O(n^2logn), n is the number of nodes in the graph
    for factor in target_factors.values():
        for node in factor.var:
            if node not in evidence and node not in out.var:
                idx = np.searchsorted(out.var, node)
                out.var = np.insert(out.var, idx, node)
                out.card = np.insert(out.card, idx, factor.card[np.where(factor.var == node)[0][0]])
    out.val = np.zeros(np.prod(out.card))

    # sample num_iterations times
    samples = []
    for _ in tqdm(range(num_iterations)):
        samples.append(_sample_step(query_nodes, proposal_factors))

    # estimate distribution with importance sampling
    # we have samples x^(1), x^(2), ..., x^(L)
    # p_tilde(x^(l)) = \prod p_u(x_u^(l) | x_pi(u)^(l))
    # q(x^(l)) = \prod q_u(x_u^(l) | x_pi(u)^(l))
    # r_tilde_l = p_tilde(x^(l)) / q(x^(l))
    # w_l = r_tilde_l / sum(r_tilde)
    # p(x_F | x_E) = \sum w_l * I(x_F^(l) = x_F) / sum(w_l)

    def cal_joint_prob(samples_with_evi, factors):
        ans = np.ones(len(samples_with_evi))
        for factor in factors.values():
            assignments = np.array(
                [[sample[node] for node in factor.var] for sample in samples_with_evi]
            ).reshape(num_iterations, -1)
            indices = np.array(
                [assignment_to_index(assignment, factor.card) for assignment in assignments]
            )
            ans *= factor.val[indices]
            # assignments.shape = (L, |var|), indices.shape = ans.shape = (L,)
        return ans
    
    samples_with_evi = [{**evidence, **sample} for sample in samples]
    p_tilde = cal_joint_prob(samples_with_evi, target_factors)
    q = cal_joint_prob(samples_with_evi, proposal_factors)
    r_tilde = p_tilde / q
    weights = r_tilde / np.sum(r_tilde)

    for sample, weight in zip(samples, weights):
        assignment = np.array([sample[node] for node in out.var]).squeeze()
        idx = assignment_to_index(assignment, out.card)
        out.val[idx] += weight
    # normalize
    out.val /= np.sum(out.val)
    """ END YOUR CODE HERE """

    return out


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
    target_factors_dict = input_config['target-factors']
    proposal_factors_dict = input_config['proposal-factors']
    assert isinstance(target_factors_dict, dict) and isinstance(proposal_factors_dict, dict)

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    target_factors = {int(node): parse_factor_dict(factor_dict=target_factor) for
                      node, target_factor in target_factors_dict.items()}
    proposal_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                        node, proposal_factor_dict in proposal_factors_dict.items()}
    evidence = input_config['evidence']
    evidence = {int(node): ev for node, ev in evidence.items()}
    num_iterations = input_config['num-iterations']
    return target_factors, proposal_factors, evidence, num_iterations


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
    target_factors, proposal_factors, evidence, num_iterations = load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(target_factors=target_factors,
                                                           proposal_factors=proposal_factors,
                                                           evidence=evidence, num_iterations=num_iterations)
    print(conditional_probability)
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
