""" CS5340 Lab 2 Part 2: Parameter Learning
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

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')  # we will store the input data files here!
OBSERVATION_DIR = os.path.join(DATA_DIR, 'observations')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """

""" END ADD HELPER FUNCTIONS HERE """


def _learn_node_parameter_w(outputs, inputs=None):
    """
    Returns the weight parameters of the linear Gaussian [w0, w1, ..., wI], where I
    is the number of inputs. Students are encouraged to use numpy.linalg.solve() to 
    get the weights. Learns weights for one node only.
    Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        inputs: N x I numpy array of input observations to the linear Gaussian model

    Returns:
        numpy array of (I + 1) weights [w0, w1, ..., wI]
    """
    num_inputs = 0 if inputs is None else inputs.shape[1] 
    weights = np.zeros(shape=num_inputs + 1)

    """ YOUR CODE HERE """
    if(num_inputs == 0): # x_u has no parents node
        weights[0] = np.mean(outputs)
        return weights

    N = inputs.shape[0]
    C = inputs.shape[1] # just num_inputs

    # C = # number of parents
    # Y = \begin{bmatrix}
    #     \sum_{n=1}^{N}{x_u,n} \\
    #     \sum_{n=1}^{N}{x_{u,n}x_{u1,n}} \\
    #     \vdots \\
    #     \sum_{n=1}^{N}{x_{u,n}x_{uC,n}}
    # \end{bmatrix}
    Y = np.zeros(shape=(C + 1, 1))
    Y[0] = np.sum(outputs)
    for i in range(1, C + 1):
        Y[i] = np.sum(outputs * inputs[:, i - 1])

    # A = \begin{bmatrix}
    #    1 & x_{u1,1} & x_{u2,1} & \cdots & x_{uC,1} \\
    #    1 & x_{u1,2} & x_{u2,2} & \cdots & x_{uC,2} \\
    #    \vdots & \vdots & \vdots & \ddots & \vdots \\
    #    1 & x_{u1,N} & x_{u2,N} & \cdots & x_{uC,N}
    # \end{bmatrix}
    A = np.ones(shape=(N, C + 1))
    A[:, 1:] = inputs

    # X = A^TA
    X = np.dot(A.T, A) # here A.shape = (N, C + 1)

    # W = (w_{\mu 0}, w_{\mu 1}, ..., w_{\mu C})^T = X^{-1}Y
    weights = np.linalg.solve(X, Y)

    """ END YOUR CODE HERE """

    return weights


def _learn_node_parameter_var(outputs, weights, inputs):
    """
    Returns the variance i.e. sigma^2 for the node. Learns variance for one node only.
    Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        weights: numpy array of (I + 1) weights of the linear Gaussian model
        inputs:  N x I numpy array of input observations to the linear Gaussian model.

    Returns:
        variance of the node's Linear Gaussian model
    """
    var = 0.

    """ YOUR CODE HERE """
    N = outputs.shape[0]
    # Z = \sum_{n=1}^N (x_{u,n} - (\sum_{c \in x_{\pi_u}} w_{uc} x_{uc,n} + w_{u0}))^2
    # var = Z / N
    Z = 0
    if(inputs is None):
        Z = np.sum(np.square(outputs - weights[0]))
    else:
        inputs = np.concatenate((np.ones(shape=(N, 1)), inputs), axis=1)
        outputs = outputs.reshape(-1, 1)
        Z = np.sum(np.square(outputs - np.dot(inputs, weights)))
    var = Z / N
    """ END YOUR CODE HERE """

    return var


def _get_learned_parameters(nodes, edges, observations):
    """
    Learns the parameters for each node in nodes and returns the parameters as a dictionary. 
    The nodes are given in ascending numerical order e.g. [1, 2, ..., V]

    Args:
        nodes: numpy array V nodes in the graph e.g. [1, 2, 3, ..., V]
        edges: numpy array of edges in the graph e.g. [i, j] implies i -> j where i is the 
            parent of j 
        observations: dictionary of node: observations pair where observations[1] 
            returns a list ofobservations for node 1.

    Returns:
        dictionary of parameters e.g.
        parameters = {
            "1": {  // first node
                "bias": w0 weight for node "1",
                "variance": variance for node "1"

                "2": weight for node "2", who is the parent of "1"
                ...
                // weights for other parents of "1"
            },
            ...
            // parameters of other nodes.
        }
    """
    parameters = {}

    """ YOUR CODE HERE """
    # Convert observations dictionary to a 2D numpy array
    observations = {node: np.array(observations[node]) for node in nodes}

    # dict for its parents, e.g. parents["1"] = ["2", "3"]
    parents = {}
    for node in nodes:
        parents[node] = []
    for edge in edges:
        parents[edge[1]].append(edge[0])
    
    for node in nodes:
        parameters[node] = {}
        inputs = np.array([observations[parent] for parent in parents[node]]).T \
            if len(parents[node]) > 0 else None
        outputs = observations[node]

        # get the weights for the node
        weights = _learn_node_parameter_w(outputs, inputs)
        # get the variance for the node
        var = _learn_node_parameter_var(outputs, weights, inputs)

        parameters[node]["bias"] = weights[0]
        parameters[node]["variance"] = var
        for i, parent in enumerate(parents[node]):
            parameters[node][parent] = weights[i + 1]

    """ END YOUR CODE HERE """

    return parameters


def main():
    """
    Helper function to load the observations, call your parameter learning function and save 
    your results. DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    observation_file = os.path.join(OBSERVATION_DIR, '{}.json'.format(case))
    with open(observation_file, 'r') as f:
         observation_config = json.load(f)

    nodes = observation_config['nodes'] # notice here `nodes` is list of strs
    edges = observation_config['edges'] # alse list of [str, str], nodes are represented as strings
    observations = observation_config['observations']

    # solution part
    parameters = _get_learned_parameters(nodes=nodes, edges=edges, observations=observations)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    for node, node_params in parameters.items():
        for param, val in node_params.items():
            node_params[param] = float(val)
        parameters[node] = node_params

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(parameters, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
