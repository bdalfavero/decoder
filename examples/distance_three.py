from typing import List
from math import floor
import numpy as np
import pandas as pd
from pathos.pools import ProcessPool
import cirq
from decoder.error_model import independent_depolarizing_model, sample_surface_error, IndependentErrorModel
from decoder.surface_decoder import decode_representative, is_error_logical_bit_flip
from decoder.ensemble_decoder import perturb_independent_model, ensemble_decode_representative

def count_errors(d: int, p: float, shots: int, model: IndependentErrorModel, ensemble: bool=False) -> int:
    """Compute the logical error rate by seeing when real logical bit flips in the sampled
    errors match up with the decoder predicting logical bit flips"""

    num_logical_errs = 0
    qs = cirq.GridQubit.rect(2 * d + 1, 2 * d + 1)
    for _ in range(shots):
        err = sample_surface_error(d, p, False)
        # Get the error class (I, X, Y, or Z) and see if this error is a logical bit flip.
        if not ensemble:
            predicted_err_class: int = decode_representative(d, err, model)
        else:
            predicted_err_class: int = ensemble_decode_representative(d, err, model, 3, 1e-4)
        error_flips_bit: bool = is_error_logical_bit_flip(d, err)
        # If the error class is I or Z, the correction flips the bit back after the error.
        err_class_flips_bit: bool = predicted_err_class == 1 or predicted_err_class == 2 
        if not err_class_flips_bit ^ (not error_flips_bit):
            num_logical_errs += 1
    return num_logical_errs


def logical_error_rate(d: int, p: float, shots: int, procs: int, model: IndependentErrorModel) -> float:
    """Compute the logical error rate by counting logical errors.
    Splits the work across a certain number of processes."""

    assert procs >= 1, "Must have at last one process."

    # Split work between the processes.
    shots_per_proc: List[int] = [floor(float(shots) / float(procs))] * procs
    leftover = shots - sum(shots_per_proc)
    for i in range(procs):
        if i < leftover:
            shots_per_proc[i] += 1
    assert sum(shots_per_proc) == shots, f"{shots_per_proc}"

    # Count logical errors across each process.
    mapper_fun = lambda this_shots: count_errors(d, p, this_shots, model)
    pool = ProcessPool(procs)
    all_ler_counts = pool.map(mapper_fun, shots_per_proc)
    return float(sum(all_ler_counts)) / float(shots)


def main():
    d = 3
    qs = cirq.GridQubit.rect(2 * d + 1, 2 * d + 1)
    shots = 1_000
    procs = 3
    ps = np.linspace(1e-3, 0.1, num=10)
    lers = []
    for p in ps:
        model = independent_depolarizing_model(qs, p)
        #model = perturb_independent_model(model, 1e-4)
        ler = logical_error_rate(d, p, shots, procs, model)
        lers.append(ler)
    df = pd.DataFrame({"p": ps, "ler": lers})
    df.set_index("p", inplace=True)
    print(df)

if __name__ == "__main__":
    main()
