"""Download Box-Jenkins data and run eT2QFNN regression."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import numpy as np

from et2qfnn import eT2QFNN


DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
DATA_FILE = Path(__file__).with_name("data") / "box_jenkins_airline_passengers.csv"


def _download_data(path: Path, refresh: bool) -> None:
    if path.exists() and not refresh:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DATA_URL, path)


def _load_passengers(path: Path) -> np.ndarray:
    values: List[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError("Unexpected CSV format in Box-Jenkins dataset")
        for row in reader:
            if len(row) < 2:
                continue
            values.append(float(row[1]))
    return np.asarray(values, dtype=float)


def _to_supervised(series: np.ndarray, lag: int) -> np.ndarray:
    if series.size <= lag:
        raise ValueError("Series too short for selected lag")
    rows = series.size - lag
    data = np.zeros((rows, lag + 1), dtype=float)
    for i in range(rows):
        data[i, :lag] = series[i : i + lag]
        data[i, lag] = series[i + lag]
    return data


def _normalize_series(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    min_val = float(np.min(series))
    max_val = float(np.max(series))
    scale = max_val - min_val
    if scale <= 0:
        scale = 1.0
    return (series - min_val) / scale, min_val, max_val


def _zscore_series(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(series))
    std = float(np.std(series))
    if std <= 0:
        std = 1.0
    return (series - mean) / std, mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eT2QFNN on Box-Jenkins data")
    parser.add_argument("--refresh", action="store_true", help="Re-download dataset")
    parser.add_argument("--lag", type=int, default=2, help="Number of lagged inputs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--mode", type=int, default=3, help="Task mode (3 = regression)")
    parser.add_argument(
        "--normalize",
        choices=("minmax", "zscore", "none"),
        default="minmax",
        help="Normalization strategy (default: minmax)",
    )
    args = parser.parse_args()

    _download_data(DATA_FILE, args.refresh)
    series = _load_passengers(DATA_FILE)

    lag = args.lag
    if lag <= 0:
        raise ValueError("Lag must be a positive integer")

    if args.normalize == "minmax":
        series, _, _ = _normalize_series(series)
    elif args.normalize == "zscore":
        series, _, _ = _zscore_series(series)

    data = _to_supervised(series, lag)

    I = lag
    N = data.shape[0]
    training_data = int(round(0.7 * N))

    eT2QFNN(data, I, args.lr, args.mode, training_data)


if __name__ == "__main__":
    main()
