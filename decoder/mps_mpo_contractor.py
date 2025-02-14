from typing import List
import jax
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator

def contract_2d_network(
    rows: int, cols: int, tn: qtn.TensorNetwork, chi: int,
    backend: str = "numpy"
) -> float:
    """Contract the 2D network corresponding to the surface code decoder.
    This is done using the MPS-MPO-MPS method as detailed in the Bravyi decoding paper.
    We treat the first and last columns as MPS's, and each intervening column as an MPO.
    Contract each of the MPO's into the leftmost MPS/column, compressing after each contraction.
    Then overlap the rightmost column."""

    assert chi >= 1, "chi must be a positive integer."
    assert rows * cols == len(tn.tensors), "Network must have the correct number of tensors."
    assert backend in ["numpy", "jax"], "Backend must be numpy or jax."

    if backend == "jax":
        for t in tn.tensors:
            t.modify(apply=lambda d: jax.numpy.array(d))

    # Check the number of tensors in each column.
    for i in range(cols):
        tag = f"col{i}"
        assert tag in tn.tag_map.keys(), "Network must have the tag col0"
        col_tensor_count = len([tn.tensors[i] for i in tn.tag_map[tag]])
        assert col_tensor_count == rows, \
            f"Column tag {tag} must have {rows} tensors, but has {col_tensor_count}."

    # Extract the first column.
    first_column_tensors = [tn.tensors[i] for i in tn.tag_map["col0"]]
    evolving_mps = qtn.TensorNetwork(first_column_tensors)
    # Contract all of the other columns in, up to the last one.
    for i in range(1, cols - 1):
        old_length: int = len(evolving_mps.tensors)
        mpo_tensors: List[qtn.Tensor] = [tn.tensors[k] for k in tn.tag_map[f"col{i}"]]
        mpo = qtn.TensorNetwork(mpo_tensors)
        evolving_mps = qtn.TensorNetwork([(evolving_mps & mpo).contract(backend=backend)]) 
        for k in range(len(evolving_mps.outer_inds())):
            evolving_mps = evolving_mps.split(evolving_mps.outer_inds()[:k], absorb="left")
        evolving_mps.compress_all(max_bond=chi, inplace=True)
    # Compute the overlap with the last column.
    last_col_tensors = [tn.tensors[i] for i in tn.tag_map[f"col{cols-1}"]]
    last_col_mps = qtn.TensorNetwork(last_col_tensors)
    return evolving_mps @ last_col_mps

if __name__ == "__main__":
    tn = qtn.TN2D_rand(3, 3, D=2, y_tag_id="col{}", x_tag_id="row{}")
    #exact_result = tn.contract()
    #my_result = contract_2d_network(3, 3, tn, 1_000_000)
    #print(exact_result - my_result)
    first_col_tensors = [tn.tensor_map[i] for i in tn.tag_map["col0"]]