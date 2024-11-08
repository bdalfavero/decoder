from math import log10
import numpy as np
import matplotlib.pyplot as plt
import stim
import pymatching
from decoder.surface_ckt import surface_code_ckt

def logical_err_rate(d: int, noise: float, shots: int, use_stim: bool=True) -> float:
    """Get logical error rate for distance and noise level.
    Does one round with no noise on the measurement qubits."""

    if use_stim:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=1,
            distance=d,
            after_clifford_depolarization=0.0,
            after_reset_flip_probability=0.0,
            before_measure_flip_probability=0.0,
            before_round_data_depolarization=noise
        )
    else:
        circuit = surface_code_ckt(d, noise)
    breakpoint()
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(shots, separate_observables=True)
    dem = circuit.detector_error_model()
    matching = pymatching.Matching.from_detector_error_model(dem)
    predictions = matching.decode_batch(detection_events)

    num_errs = 0
    for i in range(shots):
        if predictions[i] != observable_flips[i]:
            num_errs += 1
    return float(num_errs) / float(shots)


def main() -> None:
    shots = 10_000
    noise_rates = np.logspace(log10(0.1), log10(0.2), num=10)
    distances = np.array([3, 5, 7])
    lers = np.zeros((noise_rates.size, distances.size))
    for i, noise in enumerate(noise_rates):
        for j, d in enumerate(distances):
            ler = logical_err_rate(d, noise, shots, use_stim=True)
            lers[i, j] = ler
    
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(noise_rates, lers, label=distances)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()