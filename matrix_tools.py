import numpy as np
import scipy.sparse.linalg as spla
from typing import Optional, List


def block_condition_numbers(
    assembler: "pp.Assembler", A: "sps.spmatrix", variables: Optional[List[str]] = None
) -> np.ndarray:
    """ Compute condition numbers for each block in an md problem.

    Limitations / assumptions:
        1) Only subdomains (not interfaces) have condition numbers computed
        2) Zero eigenvalues are ignored in the estimation of condition numbers.
           This avoids issues relating to floating subdomains, but also ignores
           any issues with the discretization.

    Parameters:
        assembler: A PorePy assembler for the problem at hand.
        A: System matrix for the md problem
        variables: List of premissible variable names. Names not in this list
            will be ignored. Note: This is not properly tested.

    Returns:
        np.ndarray: Condition numbers for each of the matrix blocks.
    
    """

    # Relative lower bound on what will be considered a nonzero eigenvalue.
    # Eigenvalues that are less then tol (normalized by the maximum eigenvalue)
    # will be ignored in condition number estimate.
    tol = 1e-12

    def pass_all_filter(s):
        return True

    def list_filter(s):
        return s in variables

    if variables is None:
        filt = pass_all_filter
    else:
        filt = list_filter

    gb = assembler.gb

    condition_numbers = np.zeros(gb.num_graph_nodes())

    for key, block_ind in assembler.block_dof.items():
        g, var = key

        if not filt(var):
            continue
        if isinstance(g, tuple):
            # Ignore interfaces.
            continue

        # As an estimate of the number of (potentially) zero eigenvalues, find
        # the number of lower-dimensional neighbors of this grid. Use this to
        # guide how many small eigenvalues to compute. The value here may be
        # too small (in particular if used on vector problems or similar).
        num_neighs = gb.node_neighbors(g, only_lower=True).size

        ind = assembler.dof_ind(g, var)
        if ind.size == 1:
            # Condition number of a scalar is 1
            condition_numbers[block_ind] = 1
            continue
        elif ind.size < max(num_neighs + 2, 100):
            # For small problems, use dense singular value computation
            _, ev, _ = np.linalg.eig((A[ind][:, ind]).toarray())
            ev.sort()
            ev_max = ev
            ev_min = ev
        else:
            # Find largest and smallest singular values
            ev_max, _ = spla.eigs(A[ind][:, ind], k=1)
            # See comment above on the number estimated here
            ev_min, _ = spla.eigs(A[ind][:, ind], which="SM", k=2 * num_neighs + 1)
            ev_max = np.real(ev_max)
            ev_min = np.real(ev_min)
            ev_max.sort()
            ev_min.sort()

        # Find the smallest eigenvalue / singular value which is significantly
        # different from zero
        first_nonzero = np.where(ev_min > ev_max[-1] * tol)[0][0]
        condition_numbers[block_ind] = ev_max[-1] / ev_min[first_nonzero]

    return condition_numbers
