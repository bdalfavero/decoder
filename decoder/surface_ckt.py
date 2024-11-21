from typing import List, Tuple
import stim

def surface_code_ckt(d: int, p_depol: float) -> stim.Circuit:
    """Surface code circuit with phenomenological depolarizing noise."""

    # Surface code as depicted in Fig. 8a of Chubb and Flammia.
    # We will number qubits (both data and measure) left-to-right,
    # then top-to-bottom.

    qubits_per_side = 2 * d - 1

    # Store lists of qubit indices for each stabilizer, along with
    # its measurement qubit.
    plaquettes: List[Tuple[int, List[int]]] = []
    crosses: List[Tuple[int, List[int]]] = []
    for i in range(qubits_per_side): # over rows
        for j in range(qubits_per_side): # over columns
            if (i % 2 == 0) and (j % 2 != 0):
                # This is a X stabilizer (cross), blue circles in the figure.
                this_cross_idxs: List[int] = []
                if i != 0:
                    this_cross_idxs.append((i - 1) * qubits_per_side + j)
                if i != qubits_per_side - 1:
                    this_cross_idxs.append((i + 1) * qubits_per_side + j)
                if j != 0:
                    this_cross_idxs.append(i * qubits_per_side + (j - 1))
                if j != qubits_per_side - 1:
                    this_cross_idxs.append(i * qubits_per_side + (j + 1))
                crosses.append((i * qubits_per_side + j, this_cross_idxs))
            if (i % 2 != 0) and (j % 2 == 0):
                # This is a Z stabilizer (plaquette), red circles in the figure.
                this_plaq_idxs: List[int] = []
                if i != 0:
                    this_plaq_idxs.append((i - 1) * qubits_per_side + j)
                if i != qubits_per_side - 1:
                    this_plaq_idxs.append((i + 1) * qubits_per_side + j)
                if j != 0:
                    this_plaq_idxs.append(i * qubits_per_side + (j - 1))
                if j != qubits_per_side - 1:
                    this_plaq_idxs.append(i * qubits_per_side + (j + 1))
                plaquettes.append((i * qubits_per_side + j, this_plaq_idxs))
    
    ckt = stim.Circuit()
    # Initialize all qubits to zero.
    for i in range(qubits_per_side * qubits_per_side):
        ckt.append("R", i)
    # Noise before the check.
    for i in range(qubits_per_side):
        for j in range(qubits_per_side):
            if ((i % 2 == 0) and (j % 2 == 0)) or ((i % 2 != 0) and (j % 2 != 0)):
                # This is data qubit.
                ckt.append("DEPOLARIZE1", i * qubits_per_side + j, p_depol)
    # State prep circuit
    # Parity check circuit.
    for measure_qubit, plaq_qubits in plaquettes:
        for qb in plaq_qubits:
            ckt.append("CNOT", [qb, measure_qubit])
        ckt.append("MR", measure_qubit)
        ckt.append_from_stim_program_text("DETECTOR rec[-1]")
    for measure_qubit, cross_qubits in crosses:
        ckt.append("H", measure_qubit)
        for qb in cross_qubits:
            ckt.append("CNOT", [measure_qubit, qb])
        ckt.append("H", measure_qubit)
        ckt.append("MR", measure_qubit)
        ckt.append_from_stim_program_text("DETECTOR rec[-1]")
    # Logical observable measurement.
    for j in range(qubits_per_side):
        ckt.append("M", j)
    ckt.append_from_stim_program_text("OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]")
    return ckt


if __name__ == "__main__":
    ckt = surface_code_ckt(3, 0.1)