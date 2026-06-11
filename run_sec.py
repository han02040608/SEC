from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from data_utils import (
    compute_metrics,
    ensure_dir,
    inverse_scale_targets,
    prepare_experiment_data,
    save_metrics_rows,
    save_prediction_frame,
    set_global_seed,
    to_tensor_dataset,
)
from sec_model import SECModel

DEFAULT_RANDOM_SEEDS = [3407, 42, 2023, 2024, 2025, 2026, 2022, 2021, 2020, 2019]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the SEC model.")
    parser.add_argument("--data", required=True, help="Input station CSV file.")
    parser.add_argument("--split", required=True, help="Split JSON file.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and predictions.")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_RANDOM_SEEDS),
        help="Comma-separated random seeds for repeated SEC runs.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--num-transformer-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kernel-sizes", default="3,5,7")
    parser.add_argument("--amplify-ratio", type=float, default=2.0)
    return parser


def build_model(args, num_features: int) -> SECModel:
    kernel_sizes = tuple(int(v) for v in args.kernel_sizes.split(",") if v.strip())
    return SECModel(
        input_dim=num_features,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_gnn_layers=args.num_gnn_layers,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        kernel_sizes=kernel_sizes,
        amplify_ratio=args.amplify_ratio,
        max_seq_length=2048,
    )


def parse_seeds(seed_text: str) -> list[int]:
    seeds = [int(token.strip()) for token in seed_text.split(",") if token.strip()]
    if not seeds:
        raise ValueError("At least one random seed must be provided.")
    return seeds


def evaluate_split(model, x_tensor, y_tensor, scaler_y):
    model.eval()
    with torch.no_grad():
        pred_scaled, _ = model(x_tensor)
    pred = inverse_scale_targets(scaler_y, pred_scaled.detach().cpu().numpy())
    obs = inverse_scale_targets(scaler_y, y_tensor.detach().cpu().numpy())
    return obs, pred


def main() -> None:
    args = build_argparser().parse_args()
    seeds = parse_seeds(args.seeds)
    output_dir = ensure_dir(Path(args.output_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = prepare_experiment_data(args.data, args.split)
    x_train, y_train = to_tensor_dataset(exp["x_train"], exp["y_train"])
    x_val, y_val = to_tensor_dataset(exp["x_val"], exp["y_val"])
    x_test, y_test = to_tensor_dataset(exp["x_test"], exp["y_test"])

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_loader = data.DataLoader(
        data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=False,
    )

    history_rows = []
    per_seed_metrics = []
    train_pred_stack = []
    val_pred_stack = []
    test_pred_stack = []
    train_obs = val_obs = test_obs = None

    for seed in seeds:
        set_global_seed(seed)
        model = build_model(args, exp["num_features"]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_model = None
        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                pred, _ = model(batch_x)
                loss = criterion(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred_scaled, _ = model(x_val)
                val_loss = criterion(val_pred_scaled, y_val).item()

            history_rows.append(
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "train_loss": float(loss.item()),
                    "val_loss": float(val_loss),
                }
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)

        if best_model is None:
            raise RuntimeError(f"Training did not produce a valid best model for seed {seed}.")

        train_obs, train_pred = evaluate_split(best_model, x_train, y_train, exp["scaler_y"])
        val_obs, val_pred = evaluate_split(best_model, x_val, y_val, exp["scaler_y"])
        test_obs, test_pred = evaluate_split(best_model, x_test, y_test, exp["scaler_y"])

        train_pred_stack.append(train_pred)
        val_pred_stack.append(val_pred)
        test_pred_stack.append(test_pred)

        for split_name, obs, pred in (
            ("train", train_obs, train_pred),
            ("val", val_obs, val_pred),
            ("test", test_obs, test_pred),
        ):
            row = {"model": "SEC", "split": split_name, "seed": seed}
            row.update(compute_metrics(obs, pred))
            per_seed_metrics.append(row)

    train_pred_mean = np.mean(np.stack(train_pred_stack, axis=0), axis=0)
    val_pred_mean = np.mean(np.stack(val_pred_stack, axis=0), axis=0)
    test_pred_mean = np.mean(np.stack(test_pred_stack, axis=0), axis=0)

    metrics_rows = []
    for split_name, obs, pred in (
        ("train", train_obs, train_pred_mean),
        ("val", val_obs, val_pred_mean),
        ("test", test_obs, test_pred_mean),
    ):
        row = {"model": "SEC", "split": split_name, "seed": "mean_of_10_seeds"}
        row.update(compute_metrics(obs, pred))
        metrics_rows.append(row)

    save_metrics_rows(output_dir / "sec_metrics.csv", metrics_rows)
    save_metrics_rows(output_dir / "sec_metrics_by_seed.csv", per_seed_metrics)
    pd.DataFrame(history_rows).to_csv(output_dir / "sec_training_history.csv", index=False)

    save_prediction_frame(output_dir / "sec_train_predictions.csv", train_obs, train_pred_mean, exp["train_dates"], pred_column="pred_mean")
    save_prediction_frame(output_dir / "sec_val_predictions.csv", val_obs, val_pred_mean, exp["val_dates"], pred_column="pred_mean")
    save_prediction_frame(output_dir / "sec_test_predictions.csv", test_obs, test_pred_mean, exp["test_dates"], pred_column="pred_mean")


if __name__ == "__main__":
    main()
