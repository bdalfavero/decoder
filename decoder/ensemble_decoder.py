from typing import Dict
from copy import deepcopy
import numpy as np
import scipy.linalg as la
import cirq
from decoder.error_model import IndependentErrorModel
from decoder.surface_decoder import decode_representative

def perturb_independent_model(model: IndependentErrorModel, std: float, tol: float=1e-4) -> IndependentErrorModel:
    """Perturb probabilities of errrors on qubits by a Gaussian
    random variable.
    
    Arguments:
    model - An independent error model to perturb.
    std - Each probability p_I, p_X, p_Y, p_Z will have a Guassian
    random variable with the given std added to it.
    tol - We check that the probabilities for each qubit add up approximately to one."""

    new_dict: Dict[cirq.Qid, Dict[str, float]] = {}
    for q, pauli_dict in model.probability_dict.items():
        # For each qubit, generate a new set of probabilities
        # for X,Y,Z errors that are off from the original by some factor
        # epsilon.
        ps = np.array([pauli_dict["I"], pauli_dict["X"], pauli_dict["Y"], pauli_dict["Z"]])
        assert abs(1.0 - sum(ps)) <= tol, f"Probablities add to {sum(ps)}."
        gen = np.random.default_rng()
        dp = gen.normal(loc=0.0, scale=std, size=4)
        p_new = ps + dp
        p_new = p_new / la.norm(p_new)
        this_q_dict = {"I": p_new[0], "X": p_new[1], "Y": p_new[2], "Z": p_new[3]}
        new_dict[q] = deepcopy(this_q_dict)
    return IndependentErrorModel(new_dict)
