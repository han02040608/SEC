# SEC Reproducibility Package

This repository contains the SEC model implementation together with the minimal supplementary scripts needed to clarify the two benchmark details requested in review: the HBV calibration settings and the ARIMA order selection procedure.

## Files

- `sec_model.py`: SEC architecture.
- `data_utils.py`: shared preprocessing, split, windowing, scaling, and metric utilities.
- `benchmark_models.py`: ARIMA and HBV implementations used by the supplementary scripts.
- `preprocess_data.py`: clean raw station CSV files and save processed copies.
- `make_splits.py`: generate fixed train/validation/test split files.
- `run_sec.py`: train and evaluate SEC on one station split using 10 random seeds by default.
- `select_arima_order.py`: search ARIMA `(p,d,q)` using validation performance.
- `run_hbv.py`: calibrate and evaluate the HBV benchmark.
- `arima_orders.csv`: selected ARIMA orders.
- `hbv_calibration_settings.yaml`: explicit HBV calibration configuration.
- `hbv_calibrated_params.csv`: placeholder table for final calibrated HBV parameters.
- `splits/`: fixed split files.
- `example_data/`: example station data and format notes.

## Data format

Expected input is a station-level CSV file with:

- one `date` column
- meteorological feature columns such as `dayl`, `prcp`, `srad`, `tmax`, `tmin`, `vp`
- one runoff target column named `OT`

The sample file in `example_data/sample_station_data/sample_station_data.csv` follows this schema.

## Reproduction workflow

Run the commands below from the `SEC_GitHub` directory:

```bash
cd SEC_GitHub
```

1. Preprocess the raw station file.

```bash
python preprocess_data.py --input example_data/sample_station_data/sample_station_data.csv --output outputs/sample_station_processed.csv
```

2. Generate a fixed split file.

```bash
python make_splits.py --input example_data/sample_station_data/sample_station_data.csv --output splits/sample_station_data_split.json --seq-len 7 --pred-len 1
```

3. Train and evaluate SEC.

```bash
python run_sec.py --data example_data/sample_station_data/sample_station_data.csv --split splits/sample_station_data_split.json --output-dir outputs/sec_run
```

4. Search ARIMA orders if needed.

```bash
python select_arima_order.py --data example_data/sample_station_data/sample_station_data.csv --split splits/sample_station_data_split.json --output outputs/arima_order_search.csv
```

5. Calibrate and run HBV separately if needed.

```bash
python run_hbv.py --data example_data/sample_station_data/sample_station_data.csv --split splits/sample_station_data_split.json --output-dir outputs/hbv_run
```

## Notes

- All train/validation/test partitions are time-ordered and stored as explicit split files.
- `run_sec.py` uses 10 random seeds by default: `3407, 42, 2023, 2024, 2025, 2026, 2022, 2021, 2020, 2019`.
- The default SEC batch size is 512.
- `hbv_calibration_settings.yaml` documents the HBV calibration settings and bounds.
- `arima_orders.csv` documents the final selected ARIMA order for each station after search.
- Runtime output files under `outputs/` are examples only and do not need to be committed for manuscript review.
