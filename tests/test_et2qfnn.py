"""Functional test for eT2QFNN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from et2qfnn import eT2QFNN


def test_et2qfnn_runs_on_sample_data() -> None:
    data_path = Path(__file__).resolve().parents[1] / "databj.mat"
    mat = loadmat(data_path)
    data = np.asarray(mat["data"], dtype=float)

    # Keep the test fast by using a small slice, but preserve shapes.
    data = data[:60]
    N, mn = data.shape
    I = 2
    training_data = int(round(0.7 * N))
    lr = 0.01
    mode = 3

    y, rule, network, performance = eT2QFNN(
        data,
        I,
        lr,
        mode,
        training_data,
        random_state=0,
        plot=False,
        verbose=False,
    )

    assert y.shape == (N, mn - I)
    assert rule["K"] >= 1
    assert network["miu"].shape[0] == I
    assert "time" in performance
    assert np.isfinite(y).all()
