"""Run the eT2QFNN algorithm using databj.mat."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from et2qfnn import eT2QFNN


def main() -> None:
    data_path = Path(__file__).with_name("databj.mat")
    mat = loadmat(data_path)
    if "data" not in mat:
        raise KeyError("databj.mat must contain a 'data' variable")

    data = np.asarray(mat["data"], dtype=float)
    N = data.shape[0]
    training_data = int(round(0.7 * N))
    I = 2

    lr = 0.01
    mode = 3

    eT2QFNN(data, I, lr, mode, training_data)


if __name__ == "__main__":
    main()
