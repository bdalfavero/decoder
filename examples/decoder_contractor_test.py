from time import perf_counter
from typing import List
import numpy as np
import h5py
import cirq
import quimb.tensor as qtn
from decoder.surface_decoder import build_network_for_error_class
from decoder.error_model import independent_depolarizing_noise
from decoder.mps_mpo_contractor import contract_2d_network

def main() -> None:
    sizes: List[int] = [3, 5, 7, 9]
    chis: List[int] = range(2, 12)
    # Test my contractor vs. quimb's.
    # Make array of errors. Rows are network sizes, columns are bond dims.
    errs: np.ndarray = np.zeros((len(sizes), len(chis)))
    quimb_times: np.ndarray = np.zeros((len(sizes),))
    my_times: np.ndarray = np.zeros((len(sizes), len(chis)))

    for i, size in enumerate(sizes):
        rows = 2 * size - 1
        cols = 2 * size - 1
        qs = cirq.GridQubit.rect(rows, cols)
        err_qs = [qs[0], qs[1], qs[3], qs[10]]
        err = cirq.PauliString({q: cirq.X for q in err_qs})
        model = lambda e: independent_depolarizing_noise(e, 0.1)
        tn = build_network_for_error_class(qs, err, size, 0.1, model)
        quimb_start_time = perf_counter()
        result_tensor = tn.contract()
        quimb_end_time = perf_counter()
        quimb_times[i] = quimb_end_time - quimb_start_time
        for j, chi in enumerate(chis):
            my_start_time = perf_counter()
            my_result_tensor = contract_2d_network(rows, cols, tn, chi, "numpy")
            my_end_time = perf_counter()
            err = abs(my_result_tensor - result_tensor)
            errs[i, j] = err
            my_times[i, j] = my_end_time - my_start_time
    
    f = h5py.File("../data/decoder_contractor_data.hdf5", "w")
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