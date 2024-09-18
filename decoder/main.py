import numpy as np
import quimb.tensor as qtn

def main() -> None:
    a = np.array(list(range(6))).reshape(2, 3)
    a_tensor = qtn.Tensor(a, ['i', 'j'])
    print(a_tensor)

if __name__ == "__main__":
    main()
