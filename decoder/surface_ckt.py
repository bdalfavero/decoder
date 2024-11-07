from typing import List
import stim

def surface_code_ckt(d: int, p_depol: float) -> stim.Circuit:
    """Surface code circuit with phenomenological depolarizing noise."""

    # Surface code as depicted in Fig. 8a of Chubb and Flammia.
    # We will number qubits (both data and measure) left-to-right,
    # then top-to-bottom.

    

    qubits_per_side = 2 * d - 1

    # Store lists of qubit indices for each stabilizer.
    plaquettes: List[List[int]] = []
    crosses: List[List[int]] = []
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
                crosses.append(this_cross_idxs)
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
                plaquettes.append(this_plaq_idxs)
    
    ckt = stim.Circuit()
    # Noise before the check.
    for i in range(qubits_per_side):
        for j in range(qubits_per_side):
            if ((i % 2 == 0) and (j % 2 == 0)) or ((i % 2 != 0) and (j % 2 != 0)):
                # This is data qubit.
                ckt.append("DEPOLARIZE1", i * qubits_per_side + j, p_depol)
    breakpoint()
    # Parity check circuit.
    return ckt


if __name__ == "__main__":
    ckt = surface_code_ckt(3, 0.1)