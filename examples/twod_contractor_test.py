from time import perf_counter
from typing import List
import numpy as np
import h5py
import quimb.tensor as qtn
from decoder.mps_mpo_contractor import contract_2d_network

def build_network(rows: int, cols: int, chi: int) -> qtn.TensorNetwork:
    # Make a list of horizontal and vertical indices.
    horizontal_indices: List[List[str]] = []
    vertical_indices: List[List[str]] = []
    current_i: int = 0
    current_j: int = 0
    for i in range(rows): # over rows
        this_row_horizontal_inds: List[str] = []
        this_row_vertical_inds: List[str] = []
        for j in range(cols - 1): # over cols
            this_row_horizontal_inds.append(f"k{current_i}")
            this_row_vertical_inds.append(f"l{current_j}")
            current_i += 1
            current_j += 1
        # There is always one more vertical link than horizontal.
        # Compensate by adding another link.
        this_row_vertical_inds.append(f"l{current_j}")
        current_j += 1
        horizontal_indices.append(this_row_horizontal_inds)
        vertical_indices.append(this_row_vertical_inds)
    
    # Build each of the tensors.
    data_qubit_index = 0 # Track which data qubit we are currently building a tensor for.
    tensors: List[qtn.Tensor] = []
    for i in range(rows): # over rows
        for j in range(cols): # over cols
            # Get a list of the indices for this tensor.
            # These are ordered [north, south, east, west].
            inds: List[str] = []
            shape: List[int] = []
            if i != 0: # North
                inds.append(vertical_indices[i-1][j])
                shape.append(chi)
            if i != rows - 1: # South
                inds.append(vertical_indices[i][j])
                shape.append(chi)
            if j != cols - 1: # East
                inds.append(horizontal_indices[i][j])
                shape.append(chi)
            if j != 0: # East
                inds.append(horizontal_indices[i][j-1])
                shape.append(chi)
            # Build the tensor.
            tensor_data = np.random.rand(*shape) - 0.5
            tensor = qtn.Tensor(tensor_data, inds)
            tensor.add_tag(f"row{i}")
            tensor.add_tag(f"col{j}")
            #print(i, j, inds, tensor.inds)
            tensors.append(tensor)
    return qtn.TensorNetwork(tensors)


def main() -> None:
    sizes: List[int] = range(3, 7)
    chis: List[int] = range(2, 12)
    # Test my contractor vs. quimb's.
    # Make array of errors. Rows are network sizes, columns are bond dims.
    errs: np.ndarray = np.zeros((len(sizes), len(chis)))
    quimb_times: np.ndarray = np.zeros((len(sizes),))
    my_times: np.ndarray = np.zeros((len(sizes), len(chis)))

    for i, size in enumerate(sizes):
        # TODO raise the bond dim to get a compression error.
        tn = build_network(size, size, 4)
        quimb_start_time = perf_counter()
        result_tensor = tn.contract()
        quimb_end_time = perf_counter()
        quimb_times[i] = quimb_end_time - quimb_start_time
        for j, chi in enumerate(chis):
            my_start_time = perf_counter()
            my_result_tensor = contract_2d_network(size, size, tn, chi, "numpy")
            my_end_time = perf_counter()
            err = abs(my_result_tensor - result_tensor)
            errs[i, j] = err
            my_times[i, j] = my_end_time - my_start_time
    
    f = h5py.File("../data/contractor_data.hdf5", "w")
    sizes_dset = f.create_dataset("sizes", (len(sizes)), dtype=int)
    sizes_dset[:] = sizes
    chis_dset = f.create_dataset("chis", (len(chis)), dtype=int)
    chis_dset[:] = chis
    errors_dset = f.create_dataset("errors", errs.shape, dtype=float)
    errors_dset[:] = errs
    quimb_times_dset = f.create_dataset("quimb_times", quimb_times.size, dtype=float)
    quimb_times_dset[:] = quimb_times
    my_times_dset = f.create_dataset("my_times", my_times.shape, dtype=float)
    my_times_dset[:] = my_times
    f.close()


if __name__ == "__main__":
    main()