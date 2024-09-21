# taken from part 1
import copy
import numpy as np
from factor import Factor, index_to_assignment, assignment_to_index


def factor_product(A, B):
    """
    Computes the factor product of A and B e.g. A = f(x1, x2); B = f(x1, x3); 
    out=f(x1, x2, x3) = f(x1, x2)f(x1, x3)

    Args:
        A: first Factor
        B: second Factor

    Returns:
        Returns the factor product of A and B
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

    out.val = A.val[idxA] * B.val[idxB]
    return out


def factor_marginalize(factor, var):
    """
    Returns factor after variables in var have been marginalized out.

    Args:
        factor: factor to be marginalized
        var: numpy array of variables to be marginalized over

    Returns:
        marginalized factor
    """
    if factor.is_empty():
        return factor # nothing to marginalize
    out = Factor()
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


def factor_evidence(factor, evidence):
    """
    Observes evidence and retains entries containing the observed evidence. Also removes 
    the evidence random variables because they are already observed e.g. factor=f(1, 2) 
    and evidence={1: 0} returns f(2) with entries from node1=0
    Args:
        factor: factor to reduce using evidence
        evidence: dictionary of node:evidence pair where evidence[1] = evidence of node 1.
    Returns:
        Reduced factor that does not contain any variables in the evidence. Return an 
        empty factor if all the factor's variables are observed.
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE,     HINT: copy from lab2 part 1! """

    """ END YOUR CODE HERE """
    for var, value in evidence.items():
        if var in out.var:
            assignments = out.get_all_assignments()
            var_idx = np.argwhere(out.var == var)[0][0]
            indices = np.where(assignments[:, var_idx] != value)[0]
            # keep only the rows that have the evidence value
            out.val = np.delete(out.val, indices)
            # remove the variable from the factor
            out.var = np.delete(out.var, var_idx)
            out.card = np.delete(out.card, var_idx)
    return out

# A = Factor(var=np.array([1, 2, 5]), card=np.array([2, 3, 5]), val=np.arange(0, 3.1, 0.1))
# print(A)
# out = factor_evidence(A, {1: 1,  2: 0})
# print(out)


# if __name__ == '__main__':
#     main()
