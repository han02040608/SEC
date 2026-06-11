from __future__ import annotations

import argparse
from pathlib import Path

from data_utils import make_split_definition, save_split_definition


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create fixed train/val/test split files.")
    parser.add_argument("--input", required=True, help="Input station CSV file.")
    parser.add_argument("--output", required=True, help="Output split JSON file.")
    parser.add_argument("--seq-len", type=int, default=7, help="Input sequence length.")
    parser.add_argument("--pred-len", type=int, default=1, help="Forecast horizon.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Training ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    split = make_split_definition(
        csv_path=Path(args.input),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    save_split_definition(split, args.output)


if __name__ == "__main__":
    main()

