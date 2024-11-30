from typing import List
import argparse
from math import ceil
import numpy as np
import xarray as xr
import pandas as pd
from pathos.pools import ProcessPool
from decoder.surface_decoder import decode_representative, is_error_logical_bit_flip
from decoder.error_model import sample_surface_error, independent_depolarizing_noise

def count_errors(d: int, p: float, shots: int) -> int:
    """Compute the logical error rate by seeing when real logical bit flips in the sampled
    errors match up with the decoder predicting logical bit flips"""

    num_logical_errs = 0
    model = lambda e: independent_depolarizing_noise(e, p)
    for _ in range(shots):
        err = sample_surface_error(d, p, False)
        # Get the error class (I, X, Y, or Z) and see if this error is a logical bit flip.
        predicted_err_class: int = decode_representative(d, err, p, model)
        error_flips_bit: bool = is_error_logical_bit_flip(d, err)
        # If the error class is I or Z, the correction flips the bit back after the error.
        err_class_flips_bit: bool = predicted_err_class == 1 or predicted_err_class == 2 
        if not err_class_flips_bit ^ (not error_flips_bit):
            num_logical_errs += 1
    return num_logical_errs


def logical_error_rate(d: int, p: float, shots: int, procs: int) -> float:
    """Compute the logical error rate by counting logical errors.
    Splits the work across a certain number of processes."""

    assert procs >= 1, "Must have at last one process."

    # Split work between the processes.
    shots_per_proc: List[int] = [ceil(float(shots) / float(procs))] * procs
    leftover = shots - sum(shots_per_proc)
    for i in range(procs):
        if i < leftover:
            shots_per_proc[i] += 1
    assert sum(shots_per_proc) == shots

    # Count logical errors across each process.
    mapper_fun = lambda this_shots: count_errors(d, p, this_shots)
    pool = ProcessPool(procs)
    all_ler_counts = pool.map(mapper_fun, shots_per_proc)
    return float(sum(all_ler_counts)) / float(shots)


def main() -> None:    
    parser = argparse.ArgumentParser()
    parser.add_argument("shots", type=int, help="Number of shots for each 'experiment'")
    parser.add_argument("processes", type=int, help="Number of processes in the pathos pool.")
    parser.add_argument("outfile", type=str, help="Output file path.")
    args = parser.parse_args()

    shots = args.shots
    ds: np.ndarray = np.array([3, 5, 7]) # Distances
    ps: np.ndarray = np.linspace(1e-2, 0.4, num=10) # Probabilities
    lers = np.zeros((ds.size, ps.size), dtype=float)
    for i, d in enumerate(ds):
        for j, p in enumerate(ps):
            ler = logical_error_rate(d, p, shots, args.processes)
            lers[i, j] = ler
    xr_frame = xr.DataArray(lers, dims=("d", "p"), coords={"d": ds, "p": ps}, name="ler")
    pandas_df = xr_frame.to_dataframe()
    pandas_df.to_csv(args.outfile)
 
if __name__ == "__main__":
    main()