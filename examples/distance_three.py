import cirq
from decoder.error_model import independent_depolarizing_model
from decoder.surface_decoder import decode_representative

def main():
    d = 3
    qs = cirq.GridQubit.rect(2 * d + 1, 2 * d + 1)
    err = cirq.PauliString({qs[0]: cirq.X})
    model = independent_depolarizing_model(qs, 0.5)
    subclass = decode_representative(d, err, model)
    print(subclass)

if __name__ == "__main__":
    main()
