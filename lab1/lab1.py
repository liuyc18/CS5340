""" CS5340 Lab 1: Belief Propagation and Maximal Probability
See accompanying PDF for instructions.

Name: Liu Yichao
Email: e1373474@u.nus.edu
Student ID: A0304386A
"""

import copy
from typing import List

import numpy as np

from factor import Factor, index_to_assignment, assignment_to_index,\
    generate_graph_from_factors, visualize_graph

A = Factor(var=[1, 2, 5], card=[2, 3, 5], val=np.arange(0, 3, 0.1))
B = Factor(var=[2, 4], card=[3, 7], val=np.arange(0, 2.1, 0.1))
# C = Factor(var=[2, 3], card=[3, 1], val=np.arange(0, 0.3, 0.1))

"""For sum product message passing"""
def factor_product(A, B):
    """Compute product of two factors.

    Suppose A = phi(X_1, X_2), B = phi(X_2, X_3), the function should return
    phi(X_1, X_2, X_3)
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    # here the same common variable has the same cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor product
    Hint: The code for this function should be very short (~1 line). Try to
      understand what the above lines are doing, in order to implement
      subsequent parts.
    """
    out.val = A.val[idxA] * B.val[idxB]
    return out

def factor_marginalize(factor, var):
    """Sums over a list of variables.

    Args:
        factor (Factor): Input factor
        var (List): Variables to marginalize out

    Returns:
        out: Factor with variables in 'var' marginalized out.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var
    """
    # select the variables that are not in var
    out.var = np.setdiff1d(factor.var, var)
    # select the cardinality of these variables above
    out.card = factor.card[np.isin(factor.var, out.var)]
    # reshape and sum the factor values
    margins = np.where(np.isin(factor.var, var))[0]
    margins = factor.var.size - 1 - margins
    reshape_val = factor.val.reshape(factor.card[::-1])
    out.val = np.sum(reshape_val, axis=tuple(margins)).flatten()

    return out

def observe_evidence(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    if evidence is None:
        return factors
    out = copy.deepcopy(factors)

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the 
    evidence to zero.
    """
    for var, value in evidence.items():
        for factor in out:
            assignments = factor.get_all_assignments()
            if var in factor.var:
                var_idx = np.where(factor.var == var)[0][0]
                # select rows that are inconsistent with the evidence
                indices = np.where(assignments[:, var_idx] != value)[0]
                # and set to 0
                factor.val[indices] = 0
    return out

"""For max sum meessage passing (for MAP)"""
def factor_sum(A, B):
    """Same as factor_product, but sums instead of multiplies
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor sum. The code for this
    should be very similar to the factor_product().
    """
    out.val = A.val[idxA] + B.val[idxB]

    return out


def factor_max_marginalize(factor, var):
    """Marginalize over a list of variables by taking the max.

    Args:
        factor (Factor): Input factor
        var (List): Variable to marginalize out.

    Returns:
        out: Factor with variables in 'var' marginalized out. The factor's
          .val_argmax field should be a list of dictionary that keep track
          of the maximizing values of the marginalized variables.
          e.g. when out.val_argmax[i][j] = k, this means that
            when assignments of out is index_to_assignment[i],
            variable j has a maximizing value of k.
          See test_lab1.py::test_factor_max_marginalize() for an example.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var. 
    You should make use of val_argmax to keep track of the location with the
    maximum probability.
    """

    return out


def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()

    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """
    for factor in factors:
        joint = factor_product(factor, joint)

    return joint


def compute_marginals_naive(V, factors, evidence=None):
    """Computes the marginal over a set of given variables

    Args:
        V (int): Single Variable to perform inference on
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k] = v indicates that
          variable k has the value v.

    Returns:
        Factor representing the marginals
    """

    output = Factor()

    """ YOUR CODE HERE
    Compute the marginal. Output should be a factor.
    Remember to normalize the probabilities!
    """
    # compute the joint distribution
    joint = compute_joint_distribution(factors)
    # observe the evidence, reduce the joint distribution
    if evidence is not None:
        joint = observe_evidence([joint], evidence)[0]
    # marginalize out irrelevant variables
    output = factor_marginalize(joint, np.setdiff1d(joint.var, V))
    # normalize the probabilities
    output.val /= np.sum(output.val)
    return output


def compute_marginals_bp(V, factors, evidence):
    """Compute single node marginals for multiple variables
    using sum-product belief propagation algorithm

    Args:
        V (List): Variables to infer single node marginals for
        factors (List[Factor]): List of factors representing the grpahical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        marginals: List of factors. The ordering of the factors should follow
          that of V, i.e. marginals[i] should be the factor for variable V[i].
    """
    # Dummy outputs, you should overwrite this with the correct factors
    marginals = []

    # Setting up messages which will be passed
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    # Uncomment the following line to visualize the graph. Note that we create
    # an undirected graph regardless of the input graph since 1) this
    # facilitates graph traversal, and 2) the algorithm for undirected and
    # directed graphs is essentially the same for tree-like graphs.
    # visualize_graph(graph)

    # You can use any node as the root since the graph is a tree. For simplicity
    # we always use node 0 for this assignment.
    root = 0

    # Create structure to hold messages
    # A message is a list of factors.
    # Well here in a tree-like structure and after marginalization,
    # actually it's just one factor which only contains one var(receiver of the message).
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    """ YOUR CODE HERE
    Use the algorithm from lecture 4 and perform message passing over the entire
    graph. Recall the message passing protocol, that a node can only send a
    message to a neighboring node only when it has received messages from all
    its other neighbors.
    Since the provided graphical model is a tree, we can use a two-phase 
    approach. First we send messages inward from leaves towards the root.
    After this is done, we can send messages from the root node outward.
    
    Hint: You might find it useful to add auxilliary functions. You may add 
      them as either inner (nested) or external functions.
    """

    def collect(i, j):
        """
        `i`, `j` SHOULD BE NEIGHBORS in the graph, and i IS THE PARENT of j.
        We collect messages from neighbors of j except i,
        and send message from j to i.
        This message is saved in messages[j][i].
        """
        # select unary factor of j and pairwise factor between i and j
        unary_factor = graph.nodes[j]['factor'] \
            if 'factor' in graph.nodes[j] else Factor()
        pairwise_factor = graph.edges[i, j]['factor']
        # leaf node except root
        if(graph.degree(j) == 1): 
            messages[j][i] = factor_product(unary_factor, pairwise_factor)
            # sum over x_j
            messages[j][i] = factor_marginalize(messages[j][i], [j])
        # internal node
        else:
            # collect messages from neighbors of j except i
            for neighbor in list(graph.neighbors(j)):
                if neighbor != i:
                    collect(j, neighbor)
            # calculate the message from j to i
            messages[j][i] = factor_product(unary_factor, pairwise_factor)
            for neighbor in list(graph.neighbors(j)):
                if neighbor != i:
                    messages[j][i] = factor_product(messages[j][i], messages[neighbor][j])
            # sum over x_j
            messages[j][i] = factor_marginalize(messages[j][i], [j])
        return

    def distribute(i, j):
        """
        `i`, `j` SHOULD BE NEIGHBORS in the graph, and i IS THE PARENT of j.
        Distribute messages to neighbors of j except i,
        and send message from i to j.
        This message is saved in messages[i][j].
        """
        unary_factor = graph.nodes[i]['factor'] \
            if 'factor' in graph.nodes[i] else Factor()
        pairwise_factor = graph.edges[i, j]['factor']
        # calculate the message from i to j
        messages[i][j] = factor_product(unary_factor, pairwise_factor)
        for neighbor in list(graph.neighbors(i)):
            if neighbor != j:
                messages[i][j] = factor_product(messages[i][j], messages[neighbor][i])
        # sum over x_i
        messages[i][j] = factor_marginalize(messages[i][j], [i])
        # distribute messages to neighbors of j except i
        for neighbor in list(graph.neighbors(j)):
            if neighbor != i:
                distribute(j, neighbor)

    # collect the messages from leaves to root
    for neighbor in list(graph.neighbors(root)):
        collect(root, neighbor)

    # distribute the messages from root to leaves
    for neighbor in list(graph.neighbors(root)):
        distribute(root, neighbor)
    
    # print(messages)
    # compute the marginals
    for v in V:
        unary_factor = graph.nodes[v]['factor'] \
            if 'factor' in graph.nodes[v] else Factor()
        marginal = unary_factor
        for neighbor in list(graph.neighbors(v)):
            marginal = factor_product(marginal, messages[neighbor][v])
        # normalize the probabilities
        marginal.val /= np.sum(marginal.val)
        marginals.append(marginal)

    return marginals


def map_eliminate(factors, evidence):
    """Obtains the maximum a posteriori configuration for a tree graph
    given optional evidence

    Args:
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        max_decoding (Dict): MAP configuration
        log_prob_max: Log probability of MAP configuration. Note that this is
          log p(MAP, e) instead of p(MAP|e), i.e. it is the unnormalized
          representation of the conditional probability.
    """

    max_decoding = {}
    log_prob_max = 0.0

    """ YOUR CODE HERE
    Use the algorithm from lecture 5 and perform message passing over the entire
    graph to obtain the MAP configuration. Again, recall the message passing 
    protocol.
    Your code should be similar to compute_marginals_bp().
    To avoid underflow, first transform the factors in the probabilities
    to **log scale** and perform all operations on log scale instead.
    You may ignore the warning for taking log of zero, that is the desired
    behavior.
    """

    return max_decoding, log_prob_max
