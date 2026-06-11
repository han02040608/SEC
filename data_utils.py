from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler


TARGET_COLUMN = "OT"
DATE_COLUMN = "date"


@dataclass
class SplitDefinition:
    station_id: str
    csv_path: str
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seq_len: int
    pred_len: int
    target_column: str = TARGET_COLUMN
    date_column: str = DATE_COLUMN


def set_global_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_station_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f'Missing target column "{TARGET_COLUMN}" in {csv_path}.')

    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    numeric_cols = [c for c in df.columns if c != DATE_COLUMN]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(DATE_COLUMN) if DATE_COLUMN in df.columns else df.copy()
    df = df.reset_index(drop=True)
    return df


def preprocess_dataframe(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    processed = df.copy()

    numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_column not in numeric_cols:
        raise ValueError(f'Target column "{target_column}" must be numeric.')

    processed[numeric_cols] = processed[numeric_cols].replace([np.inf, -np.inf], np.nan)
    processed[numeric_cols] = processed[numeric_cols].interpolate(
        method="linear", limit_direction="both"
    )
    processed[numeric_cols] = processed[numeric_cols].fillna(
        processed[numeric_cols].median(numeric_only=True)
    )
    return processed


def get_feature_columns(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> List[str]:
    return [c for c in df.columns if c not in {DATE_COLUMN, target_column}]


def make_split_definition(
    csv_path: str | Path,
    seq_len: int,
    pred_len: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SplitDefinition:
    csv_path = Path(csv_path)
    df = preprocess_dataframe(load_station_dataframe(csv_path))
    total = len(df)
    if total <= seq_len + pred_len:
        raise ValueError("Dataset is too short for the requested sequence length.")

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_end = max(seq_len + pred_len, train_end)
    val_end = max(train_end + 1, val_end)
    val_end = min(val_end, total - 1)

    return SplitDefinition(
        station_id=csv_path.stem,
        csv_path=str(csv_path.as_posix()),
        train_start=0,
        train_end=train_end,
        val_start=train_end,
        val_end=val_end,
        test_start=val_end,
        test_end=total,
        train_ratio=train_end / total,
        val_ratio=(val_end - train_end) / total,
        test_ratio=(total - val_end) / total,
        seq_len=int(seq_len),
        pred_len=int(pred_len),
    )


def save_split_definition(split: SplitDefinition, output_path: str | Path) -> None:
    output = Path(output_path)
    ensure_dir(output.parent)
    output.write_text(json.dumps(asdict(split), indent=2), encoding="utf-8")


def load_split_definition(path: str | Path) -> SplitDefinition:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitDefinition(**data)


def split_dataframe(
    df: pd.DataFrame, split: SplitDefinition
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.iloc[split.train_start : split.train_end].reset_index(drop=True)
    val_df = df.iloc[split.val_start : split.val_end].reset_index(drop=True)
    test_df = df.iloc[split.test_start : split.test_end].reset_index(drop=True)
    return train_df, val_df, test_df


def _window_target(values: np.ndarray, seq_len: int, pred_len: int) -> np.ndarray:
    offset = seq_len + pred_len - 1
    return values[offset:]


def build_windows(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    seq_len: int,
    pred_len: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_values = np.asarray(feature_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float).reshape(-1, 1)

    num_samples = feature_values.shape[0] - seq_len - pred_len + 1
    if num_samples <= 0:
        raise ValueError("Split is too short to form one training window.")

    feature_windows = sliding_window_view(
        feature_values, window_shape=seq_len, axis=0
    )[:num_samples]
    feature_windows = np.transpose(feature_windows, (0, 2, 1))
    targets = _window_target(target_values, seq_len, pred_len)[:num_samples]
    return feature_windows, targets


def to_tensor_dataset(
    features: np.ndarray, targets: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    return x, y


def prepare_experiment_data(
    csv_path: str | Path,
    split_path: str | Path,
    target_column: str = TARGET_COLUMN,
) -> Dict[str, Any]:
    split = load_split_definition(split_path)
    df = preprocess_dataframe(load_station_dataframe(csv_path), target_column=target_column)
    feature_columns = get_feature_columns(df, target_column=target_column)
    train_df, val_df, test_df = split_dataframe(df, split)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_features_raw = train_df[feature_columns].to_numpy(dtype=float)
    val_features_raw = val_df[feature_columns].to_numpy(dtype=float)
    test_features_raw = test_df[feature_columns].to_numpy(dtype=float)

    train_target_raw = train_df[[target_column]].to_numpy(dtype=float)
    val_target_raw = val_df[[target_column]].to_numpy(dtype=float)
    test_target_raw = test_df[[target_column]].to_numpy(dtype=float)

    scaler_x.fit(train_features_raw)
    scaler_y.fit(train_target_raw)

    train_features = scaler_x.transform(train_features_raw)
    val_features = scaler_x.transform(val_features_raw)
    test_features = scaler_x.transform(test_features_raw)

    train_target = scaler_y.transform(train_target_raw)
    val_target = scaler_y.transform(val_target_raw)
    test_target = scaler_y.transform(test_target_raw)

    x_train, y_train = build_windows(train_features, train_target, split.seq_len, split.pred_len)
    x_val, y_val = build_windows(val_features, val_target, split.seq_len, split.pred_len)
    x_test, y_test = build_windows(test_features, test_target, split.seq_len, split.pred_len)

    observed_train = scaler_y.inverse_transform(y_train)
    observed_val = scaler_y.inverse_transform(y_val)
    observed_test = scaler_y.inverse_transform(y_test)

    train_dates = train_df[DATE_COLUMN].iloc[split.seq_len + split.pred_len - 1 :].reset_index(drop=True) if DATE_COLUMN in train_df.columns else None
    val_dates = val_df[DATE_COLUMN].iloc[split.seq_len + split.pred_len - 1 :].reset_index(drop=True) if DATE_COLUMN in val_df.columns else None
    test_dates = test_df[DATE_COLUMN].iloc[split.seq_len + split.pred_len - 1 :].reset_index(drop=True) if DATE_COLUMN in test_df.columns else None

    return {
        "split": split,
        "dataframe": df,
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "train_features_raw": train_features_raw,
        "val_features_raw": val_features_raw,
        "test_features_raw": test_features_raw,
        "train_target_raw": train_target_raw.reshape(-1),
        "val_target_raw": val_target_raw.reshape(-1),
        "test_target_raw": test_target_raw.reshape(-1),
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "observed_train": observed_train,
        "observed_val": observed_val,
        "observed_test": observed_test,
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
    }


def inverse_scale_targets(scaler_y: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    return scaler_y.inverse_transform(values)


def align_prediction_to_observed(predicted: np.ndarray, target_len: int) -> np.ndarray:
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    if pred.size >= target_len:
        pred = pred[-target_len:]
    else:
        pad_value = pred[0] if pred.size else 0.0
        pred = np.concatenate([np.full(target_len - pred.size, pad_value, dtype=float), pred])
    return pred.reshape(-1, 1)


def compute_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    obs = np.asarray(observed, dtype=float).reshape(-1)
    pred = np.asarray(predicted, dtype=float).reshape(-1)

    mae = float(np.mean(np.abs(obs - pred)))
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))

    denom = np.sum((obs - np.mean(obs)) ** 2)
    nse = float(1.0 - np.sum((obs - pred) ** 2) / denom) if denom > 0 else float("nan")

    obs_std = float(np.std(obs))
    pred_std = float(np.std(pred))
    if obs_std > 0 and pred_std > 0:
        corr = float(np.corrcoef(obs, pred)[0, 1])
    else:
        corr = 0.0

    alpha = pred_std / obs_std if obs_std > 0 else float("nan")
    beta = float(np.mean(pred) / np.mean(obs)) if np.mean(obs) != 0 else float("nan")
    kge = float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    return {
        "NSE": nse,
        "KGE": kge,
        "MAE": mae,
        "RMSE": rmse,
        "Pearson": corr,
    }


def save_prediction_frame(
    output_path: str | Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    dates: Optional[pd.Series] = None,
    pred_column: str = "pred",
) -> None:
    df = pd.DataFrame({"real": np.asarray(observed).reshape(-1), pred_column: np.asarray(predicted).reshape(-1)})
    if dates is not None and len(dates) == len(df):
        df.insert(0, DATE_COLUMN, pd.to_datetime(dates).dt.strftime("%Y-%m-%d"))
    output = Path(output_path)
    ensure_dir(output.parent)
    if output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    else:
        df.to_excel(output, index=False)


def save_metrics_rows(output_path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    output = Path(output_path)
    ensure_dir(output.parent)
    pd.DataFrame(rows).to_csv(output, index=False)
