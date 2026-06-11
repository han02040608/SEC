# SEC Reproducibility Package

This repository contains the SEC model implementation together with the supplementary scripts and files needed to clarify reproducibility, especially the ARIMA order selection procedure and the HBV calibration settings mentioned in review.

## Repository contents

- `sec_model.py`: SEC architecture.
- `run_sec.py`: train and evaluate SEC on one station split. By default, it uses 10 random seeds and batch size 512.
- `data_utils.py`: shared preprocessing, split, scaling, windowing, and metric utilities.
- `preprocess_data.py`: preprocess raw station data.
- `make_splits.py`: generate fixed train/validation/test split files.
- `benchmark_models.py`: ARIMA and HBV implementations used by the supplementary scripts.
- `select_arima_order.py`: ARIMA order selection script.
- `run_hbv.py`: HBV calibration and evaluation script.
- `arima_orders.csv`: final selected ARIMA orders.
- `hbv_calibration_settings.yaml`: explicit HBV calibration settings.
- `hbv_calibrated_params.csv`: calibrated HBV parameter table.
- `splits/`: fixed train/validation/test split files.
- `example_data/`: example data and format notes.

## Data format

Expected input is a station-level CSV file containing:

- one `date` column
- meteorological feature columns such as `dayl`, `prcp`, `srad`, `tmax`, `tmin`, and `vp`
- one runoff target column named `OT`

The sample file in `example_data/sample_station_data/sample_station_data.csv` follows this format.

## Reproducibility workflow

Run the commands below from the repository root.

1. Preprocess the raw station file.

```bash
python preprocess_data.py --input example_data/sample_station_data/sample_station_data.csv --output outputs/sample_station_processed.csv
```

2. Generate or inspect the fixed split file.

```bash
python make_splits.py --input example_data/sample_station_data/sample_station_data.csv --output splits/sample_station_data_split.json --seq-len 7 --pred-len 1
```

3. Train and evaluate SEC.

```bash
python run_sec.py --data example_data/sample_station_data/sample_station_data.csv --split splits/sample_station_data_split.json --output-dir outputs/sec_run
```

4. Reproduce the ARIMA order selection procedure.

```bash
python select_arima_order.py --data example_data/sample_station_data/sample_station_data.csv --split splits/sample_station_data_split.json --output outputs/arima_order_search.csv
```

5. Reproduce the HBV calibration procedure.

```bash
python run_hbv.py --data example_data/sample_station_data/sample_station_data.csv --split splits/sample_station_data_split.json --output-dir outputs/hbv_run
```

## Notes

- All train/validation/test partitions are time-ordered and stored explicitly in `splits/`.
- `run_sec.py` uses the following 10 random seeds by default: `3407, 42, 2023, 2024, 2025, 2026, 2022, 2021, 2020, 2019`.
- The default SEC batch size is 512.
- `arima_orders.csv` records the final ARIMA order used for each station after order search.
- `hbv_calibration_settings.yaml` documents the HBV objective function, warm-up setting, parameter bounds, and optimization settings.
- `hbv_calibrated_params.csv` is used to report the final calibrated HBV parameters.
- Runtime output files under `outputs/` are not required for repository submission.
