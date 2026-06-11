from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_models import ARIMABenchmark
from data_utils import align_prediction_to_observed, compute_metrics, ensure_dir, prepare_experiment_data


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select ARIMA order using validation performance.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--p-range", default="0,1,2,3,4,5")
    parser.add_argument("--d-range", default="0,1")
    parser.add_argument("--q-range", default="0,1,2,3")
    parser.add_argument("--fit-window", type=int, default=730)
    parser.add_argument("--refit-interval", type=int, default=10)
    return parser


def parse_range(text: str):
    return [int(v) for v in text.split(",") if v.strip()]


def main() -> None:
    args = build_argparser().parse_args()
    exp = prepare_experiment_data(args.data, args.split)
    rows = []

    for p in parse_range(args.p_range):
        for d in parse_range(args.d_range):
            for q in parse_range(args.q_range):
                model = ARIMABenchmark(
                    fit_window=args.fit_window,
                    refit_interval=args.refit_interval,
                    order=(p, d, q),
                )
                _, val_pred, _, _ = model.fit_predict(
                    exp["train_target_raw"], exp["val_target_raw"], exp["test_target_raw"]
                )
                metrics = compute_metrics(
                    exp["observed_val"],
                    align_prediction_to_observed(val_pred, len(exp["observed_val"])),
                )
                row = {"p": p, "d": d, "q": q}
                row.update(metrics)
                rows.append(row)

    df = pd.DataFrame(rows).sort_values(["MAE", "RMSE", "q", "p", "d"], ascending=[True, True, True, True, True])
    output = Path(args.output)
    ensure_dir(output.parent)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
