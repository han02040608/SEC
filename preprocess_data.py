from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_utils import ensure_dir, load_station_dataframe, preprocess_dataframe


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess SEC station data.")
    parser.add_argument("--input", required=True, help="Input station CSV file.")
    parser.add_argument("--output", required=True, help="Output cleaned CSV file.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    df = load_station_dataframe(args.input)
    processed = preprocess_dataframe(df)
    output = Path(args.output)
    ensure_dir(output.parent)
    processed.to_csv(output, index=False)

    summary = pd.DataFrame(
        {
            "column": processed.columns,
            "dtype": [str(t) for t in processed.dtypes],
            "missing_after_preprocess": processed.isna().sum().tolist(),
        }
    )
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)


if __name__ == "__main__":
    main()

