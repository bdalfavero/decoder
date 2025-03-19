import numpy as np
import stim
import pymatching
from decoder.dem_decoder import DEMDecoder

def five_qubit_ckt(p: float) -> stim.Circuit:
    """Get memory experiment circuit for the five qubit code
    with depolarizing noise of strength p on the data qubits
    (code capacity error model)."""

    assert p >= 0.0 and p <= 1.0, f"p must be a probability, but is {p}."

    data_qs = list(range(5))
    measure_qs = list(range(5, 9))

    ckt = stim.Circuit()
    # Prepare the logical |0> state.
    # See De and Pryadko arXiv:509.01239v1
    for q in sorted(data_qs + measure_qs):
        ckt.append("R", q)
    for q in range(1, len(data_qs)):
        ckt.append("H", q)
    for q in range(1, len(data_qs)):
        ckt.append("CX", [q, data_qs[0]])
    for q in range(len(data_qs)):
        ckt.append("CZ", [q, (q + 1) % len(data_qs)])

    # Subject the data qubits to depolarizing noise.
    for dq in data_qs:
        ckt.append("DEPOLARIZE1", dq, arg=p)

    # Do a parity check.
    for mq in measure_qs:
        ckt.append("H", mq)
    # First stabilizer XZZXI
    ckt.append("CX", [measure_qs[0], data_qs[0]])
    ckt.append("CZ", [measure_qs[0], data_qs[1]])
    ckt.append("CZ", [measure_qs[0], data_qs[2]])
    ckt.append("CX", [measure_qs[0], data_qs[3]])
    # Second stabilizer IXZZX
    ckt.append("CX", [measure_qs[1], data_qs[1]])
    ckt.append("CZ", [measure_qs[1], data_qs[2]])
    ckt.append("CZ", [measure_qs[1], data_qs[3]])
    ckt.append("CX", [measure_qs[1], data_qs[4]])
    # Third stabilizer XIXZZ
    ckt.append("CX", [measure_qs[2], data_qs[0]])
    ckt.append("CX", [measure_qs[2], data_qs[2]])
    ckt.append("CZ", [measure_qs[2], data_qs[3]])
    ckt.append("CZ", [measure_qs[2], data_qs[4]])
    # Third stabilizer ZXIXZ
    ckt.append("CZ", [measure_qs[3], data_qs[0]])
    ckt.append("CX", [measure_qs[3], data_qs[1]])
    ckt.append("CX", [measure_qs[3], data_qs[3]])
    ckt.append("CZ", [measure_qs[3], data_qs[4]])
    for mq in measure_qs:
        ckt.append("H", mq)
    for mq in measure_qs:
        ckt.append("MR", mq)
    for i in range(len(measure_qs)):
        ckt.append_from_stim_program_text(f"DETECTOR rec[{-1 - i}]")

    # Do a logical Z measurement.
    for dq in data_qs:
        ckt.append("M", dq)
    ckt.append_from_stim_program_text("OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3] rec[-4] rec[-5]")
    return ckt


def test_ideal_ckt():
    """Test the circuit when there is no noise present."""

    ckt = five_qubit_ckt(0.0)
    sampler = ckt.compile_sampler()
    samples = sampler.sample(shots=10_000)
    for i in range(samples.shape[0]):
        syndromes = samples[i, :][:4]
        assert len(syndromes) == 4
        logicals = samples[i, :][4:]
        assert len(logicals) == 5
        assert all(np.invert(syndromes))
        assert not bool(sum(logicals) % 2)


def pymatching_callback(dem, detector_events) -> np.ndarray:
    """Use pymatching to decode detector events."""

    matcher = pymatching.Matching.from_detector_error_model(dem)
    predictions = [bool(b) for b in matcher.decode_batch(detector_events)]
    return predictions


def dem_decoder_callback(dem, syndrome_events, p) -> np.ndarray:
    """Use Chubb's detector picture decoder on the syndromes."""

    # Prepare 5 qubit code deocder.
    stabilizers = []
    stabilizers.append(stim.PauliString("+XZZXI"))
    stabilizers.append(stim.PauliString("+IXZZX"))
    stabilizers.append(stim.PauliString("+XIXZZ"))
    stabilizers.append(stim.PauliString("+ZXIXZ"))

    logicals = []
    logicals.append(stim.PauliString("+XZIZI"))
    logicals.append(stim.PauliString("+IZIXX"))

    # Give each qubit depolarizing noise.
    single_q_prob = np.array([1.0 - p, p / 3.0, p / 3.0, p / 3.0])
    probabilities = np.zeros((5, 4), dtype=float)
    for i in range(5):
        probabilities[i, :] = single_q_prob

    decoder = DEMDecoder(probabilities, stabilizers, logicals)
    classes =  decoder.decode_batch(syndrome_events)

    flips: List[bool] = []
    for err_class in classes:
        if err_class == 1 or err_class == 2:
            flips.append(True)
        else:
            flips.append(False)
    return np.array(flips)


def count_errors(p: float, shots: int, decoder_callback, use_syndrome=False) -> int:
    """Simulate shots of the memory experiment, and count how many times
    pymatching's prediction of the observable flip does not match the observed
    observable flip.
    
    Arguments:
    p - depolarizing error probability.
    shots - number of shots to simulate
    decoder_callback - callback mapping DEM and either syndromes or detectors to predictions.
    use_syndrome - If true, uses syndromes for decoding. Otherwise, uses detectors."""

    ckt = five_qubit_ckt(p)
    dem = ckt.detector_error_model()
    if use_syndrome:
        detector_sampler = ckt.compile_detector_sampler()
        measurement_sampler = ckt.compile_sampler()
        detector_flips, observable_flips = detector_sampler.sample(shots=shots, separate_observables=True)
        all_samples = measurement_sampler.sample(shots=shots)
        syndrome_samples = all_samples[:, :4] # Only the first four columns are the syndrome.
        predictions = decoder_callback(dem, syndrome_samples)
    else:
        detector_sampler = ckt.compile_detector_sampler()
        detector_flips, observable_flips = detector_sampler.sample(shots=shots, separate_observables=True)
        predictions = decoder_callback(dem, detector_flips)

    num_errors = 0
    for s in range(shots):
        if predictions[s] != observable_flips[s]:
            num_errors += 1
    return num_errors

if __name__ == "__main__":
    shots = 1000
    ps = np.linspace(1e-3, 1e-2, num=10)
    pymatching_lers = []
    dem_lers = []
    for p in ps:
        pymatching_num_errors = count_errors(p, shots, pymatching_callback)
        pymatching_lers.append(float(pymatching_num_errors) / float(shots))
        dem_num_errors = count_errors(
            p, shots,
            lambda dem, syndrome: dem_decoder_callback(dem, syndrome, p),
            use_syndrome=True
        )
        dem_lers.append(float(dem_num_errors) / float(shots))
    print(pymatching_lers)
    print(dem_lers)
