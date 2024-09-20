from copy import deepcopy
import cupy
from cuquantum import contract
from cuquantum.cutensornet.tensor import decompose, SVDMethod
from typing import List, Tuple

def tensor_to_mps(tensor: cupy.ndarray, inds: str) -> List[cupy.ndarray]:
    """Decompose the given tensor into an MPS."""

    assert len(tensor.shape) == len(inds), "Index and tensor shape must match."
    output_tensors = []
    output_inds = []
    outer_indices = inds
    leftover_tensor = deepcopy(tensor)
    for i in range(len(tensor.shape) - 1):
        left_ind = outer_indices[i]
        right_inds = outer_indices[(i+1):]
        # Create a string describing indices in the decomposition.
        if i == 0:
            descriptor = left_ind + ''.join(right_inds) + '->' \
                + left_ind + 'A,A' + ''.join(right_inds)
        else:
            descriptor = 'B' + left_ind + ''.join(right_inds) + '->' \
                + 'B' + left_ind + 'A,A' + ''.join(right_inds)
        u, s, vdag = decompose(descriptor, leftover_tensor, method=SVDMethod())
        leftover_tensor = contract('AB,B' + ''.join(right_inds) + '->A' + ''.join(right_inds), \
                                   cupy.diag(s), vdag)
        output_tensors.append(u)
        if i != 0:
            output_inds.append(chr(ord('A') + i - 1) + outer_indices[i] + chr(ord('A') + i))
        else:
            output_inds.append(outer_indices[i] + chr(ord('A') + i))
    output_tensors.append(leftover_tensor)
    output_inds.append(chr(ord('A') + len(tensor.shape) - 2) + outer_indices[-1])
    return output_tensors, output_inds


def get_free_indices(inds: List[str]) -> str:
    """Get the free indices in order to contract a tensor network.
    
    Arguments:
    inds - a list of strings with indices of each tensor to be contracted, e.g.
    ['ij', 'jk', 'mn']
    n.b. the indices of the resulting tensor should not be here!
    
    Returns:
    free_inds - string with the non-repeated indices from the spec.
    For the example above, we should return 'ikmn'."""

    joined_str = ''.join(inds)
    for c in joined_str:
        if joined_str.count(c) > 1:
            joined_str = joined_str.replace(c, '')
    return joined_str


def sweepline_contract(
    tensors: List[cupy.ndarray],
    description: List[str]
) -> Tuple[str, cupy.ndarray]:
    """Contract a tensor network using the sweepline algorithm
    
    Arguments:
    tensors - a list of tensors to contract together.
    description - a list of strings containing the indices of the tensors,
        e.g. ['abc', 'cde'] will result in a tensor with indices 'abde'
    
    Returns:
    indices - indices of the resulting tensor, e.g. 'abde'.
    tensor - tensor that results from the contraction."""

    for i, (ten, ind) in enumerate(zip(tensors, description)):
        if i == 0:
            mps_tensors, mps_inds = tensor_to_mps(ten, ind)
        else:
            # Find the list of free indices.
            free_inds = get_free_indices(mps_inds + [ind])
            descriptor = ','.join(mps_inds + [ind]) + '->' + free_inds
            contracted_tensor = contract(descriptor, *mps_tensors, ten)
            # Then decompose the resulting tensor into an MPS.
            if i != len(tensors) - 1:
                mps_tensors, mps_inds = tensor_to_mps(contracted_tensor, free_inds)
    return free_inds, contracted_tensor