from time import perf_counter
import numpy as np
import h5py
import quimb.tensor as qtn
from decoder.mps_mpo_contractor import contract_2d_network

def main():
    ls = range(2, 8)
    chis = range(2, 8)
    quimb_times = np.zeros(len(ls))
    my_times = np.zeros((len(ls), len(chis)))
    for i, l in enumerate(ls):
        tn = qtn.TN2D_rand(l, l, D=4, y_tag_id="col{}", x_tag_id="row{}")
        quimb_start_time = perf_counter()
        tn.contract()
        quimb_end_time = perf_counter()
        quimb_times[i] = quimb_end_time - quimb_start_time
        for j, chi in enumerate(chis):
            my_start_time = perf_counter()
            contract_2d_network(l, l, tn, chi)
            my_end_time = perf_counter()
            my_times[i, j] = my_end_time - my_start_time
    f = h5py.File("../data/size_scaling.hdf5", "w")
    ls_dset = f.create_dataset("l", (len(ls),), dtype=int)
    ls_dset[:] = np.array(ls)
    chis_dset = f.create_dataset("chi", (len(list(chis)),), dtype=int)
    chis_dset[:] = np.array(chis)
    quimb_times_dset = f.create_dataset("quimb_times", quimb_times.shape, dtype=float)
    quimb_times_dset[:] = quimb_times
    my_times_dset = f.create_dataset("my_times", my_times.shape, dtype=float)
    my_times_dset[:, :] = my_times
    f.close()

if __name__ == "__main__":
    main()