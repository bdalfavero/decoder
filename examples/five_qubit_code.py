import numpy as np
import stim
import pymatching

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

if __name__ == "__main__":
    ckt = five_qubit_ckt(0.1)
    detector_sampler = ckt.compile_detector_sampler()
    detector_flips, observable_flips = detector_sampler.sample(shots=10, separate_observables=True)
    dem = ckt.detector_error_model()
    matcher = pymatching.Matching.from_detector_error_model(dem)
    predictions = [bool(b) for b in matcher.decode_batch(detector_flips)]
    print(predictions, observable_flips[:, 0])
