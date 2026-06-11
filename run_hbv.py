from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmark_models import HBVBenchmark
from data_utils import (
    align_prediction_to_observed,
    compute_metrics,
    ensure_dir,
    prepare_experiment_data,
    save_prediction_frame,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HBV calibration and evaluation.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--maxiter", type=int, default=300)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    exp = prepare_experiment_data(args.data, args.split)
    hbv = HBVBenchmark(warmup=args.warmup, maxiter=args.maxiter)

    train_forcing = HBVBenchmark.prepare_forcing(exp["train_features_raw"], exp["feature_columns"])
    val_forcing = HBVBenchmark.prepare_forcing(exp["val_features_raw"], exp["feature_columns"])
    test_forcing = HBVBenchmark.prepare_forcing(exp["test_features_raw"], exp["feature_columns"])

    train_pred, val_pred, test_pred, info = hbv.fit_predict(
        train_forcing=train_forcing,
        val_forcing=val_forcing,
        test_forcing=test_forcing,
        train_runoff=exp["train_target_raw"],
        val_runoff=exp["val_target_raw"],
    )

    pd.DataFrame([info]).to_csv(output_dir / "hbv_run_info.csv", index=False)

    rows = []
    for split_name, obs, pred, dates in (
        ("train", exp["observed_train"], train_pred, exp["train_dates"]),
        ("val", exp["observed_val"], val_pred, exp["val_dates"]),
        ("test", exp["observed_test"], test_pred, exp["test_dates"]),
    ):
        pred_2d = align_prediction_to_observed(pred, len(obs))
        row = {"model": "HBV", "split": split_name}
        row.update(compute_metrics(obs, pred_2d))
        rows.append(row)
        save_prediction_frame(output_dir / f"hbv_{split_name}_predictions.csv", obs, pred_2d, dates)

    pd.DataFrame(rows).to_csv(output_dir / "hbv_metrics.csv", index=False)


if __name__ == "__main__":
    main()
