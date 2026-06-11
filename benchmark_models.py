from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None


class PersistenceBenchmark:
    supports_random_seed = False

    def fit_predict(self, train_series, val_series, test_series):
        train_series = np.asarray(train_series, dtype=float).reshape(-1)
        val_series = np.asarray(val_series, dtype=float).reshape(-1)
        test_series = np.asarray(test_series, dtype=float).reshape(-1)
        pred_train = np.concatenate([[train_series[0]], train_series[:-1]])
        pred_val = np.full_like(val_series, train_series[-1])
        history_last = val_series[-1] if val_series.size else train_series[-1]
        pred_test = np.full_like(test_series, history_last)
        return pred_train, pred_val, pred_test, {
            "method": "Persistence",
            "fit_time_s": 0.0,
            "inference_time_s": 0.0,
        }


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x_pad = torch.cat([front, x, end], dim=1).transpose(1, 2)
        return self.avg(x_pad).transpose(1, 2)


class DLinearModel(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int, kernel_size: int = 3):
        super().__init__()
        self.decomposition = MovingAvg(kernel_size=kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.output_head = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor):
        trend = self.decomposition(x)
        seasonal = x - trend
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)
        x_out = self.linear_seasonal(seasonal) + self.linear_trend(trend)
        x_last = x_out[:, :, -1]
        return self.output_head(x_last), None


class ARIMABenchmark:
    supports_random_seed = False

    def __init__(self, fit_window: int = 730, refit_interval: int = 10, order=(5, 1, 0), use_log: bool = True):
        self.fit_window = int(fit_window)
        self.refit_interval = int(refit_interval)
        self.order = tuple(order)
        self.use_log = bool(use_log)

    @staticmethod
    def _safe_1d(arr):
        arr = np.asarray(arr, dtype=float).reshape(-1)
        return arr if arr.size else np.array([0.0], dtype=float)

    def _fit_arima_or_fallback(self, series):
        series = self._safe_1d(series)
        transformed = np.log1p(np.maximum(series, 0.0)) if self.use_log else series

        if ARIMA is not None and transformed.size >= sum(self.order) + 2:
            try:
                fit = ARIMA(transformed, order=self.order).fit()
                fitted = np.asarray(fit.fittedvalues, dtype=float).reshape(-1)
                if self.use_log:
                    fitted = np.expm1(fitted)

                def forecast_fn(steps):
                    fcast = np.asarray(fit.forecast(steps=int(steps)), dtype=float).reshape(-1)
                    return np.expm1(fcast) if self.use_log else fcast

                return fitted, forecast_fn
            except Exception:
                pass

        last_val = float(series[-1])
        fitted = np.full(series.size, last_val, dtype=float)
        return fitted, lambda steps: np.full(int(steps), last_val, dtype=float)

    def _walk_forward_predict(self, history_series, target_series):
        history = self._safe_1d(history_series).tolist()
        target = self._safe_1d(target_series)
        preds = np.zeros(target.size, dtype=float)
        fit_time = 0.0
        infer_time = 0.0
        current_forecast_fn = None
        last_fit_idx = -1

        for i in range(target.size):
            need_refit = current_forecast_fn is None or (
                self.refit_interval > 0 and (i - last_fit_idx) >= self.refit_interval
            )
            if need_refit:
                fit_series = np.asarray(history[-self.fit_window :], dtype=float) if self.fit_window > 0 else np.asarray(history, dtype=float)
                t0 = time.perf_counter()
                _, current_forecast_fn = self._fit_arima_or_fallback(fit_series)
                fit_time += time.perf_counter() - t0
                last_fit_idx = i

            t0 = time.perf_counter()
            steps_ahead = i - last_fit_idx + 1
            forecast = np.asarray(current_forecast_fn(steps_ahead), dtype=float).reshape(-1)
            infer_time += time.perf_counter() - t0
            preds[i] = forecast[-1] if forecast.size else history[-1]
            history.append(float(target[i]))

        return preds, fit_time, infer_time

    @staticmethod
    def _align(pred, target_len):
        pred = np.asarray(pred, dtype=float).reshape(-1)
        if pred.size >= target_len:
            return pred[-target_len:]
        pad = np.full(target_len - pred.size, pred[0] if pred.size else 0.0)
        return np.concatenate([pad, pred], axis=0)

    def fit_predict(self, train_series, val_series, test_series):
        train = self._safe_1d(train_series)
        val = self._safe_1d(val_series)
        test = self._safe_1d(test_series)

        fit_start = time.perf_counter()
        pred_train, _ = self._fit_arima_or_fallback(train)
        fit_time = time.perf_counter() - fit_start

        pred_val, val_fit_time, val_infer_time = self._walk_forward_predict(train, val)
        pred_test, test_fit_time, test_infer_time = self._walk_forward_predict(
            np.concatenate([train, val]), test
        )
        fit_time += val_fit_time + test_fit_time

        return (
            self._align(pred_train, train.size),
            self._align(pred_val, val.size),
            self._align(pred_test, test.size),
            {
                "method": "ARIMA",
                "fit_time_s": float(fit_time),
                "inference_time_s": float(val_infer_time + test_infer_time),
                "order": list(self.order),
            },
        )


@dataclass
class HBVConfig:
    warmup: int = 30
    maxiter: int = 300


class HBVBenchmark:
    supports_random_seed = False

    def __init__(self, warmup: int = 30, maxiter: int = 300):
        self.warmup = int(warmup)
        self.maxiter = int(maxiter)
        self.bounds = [
            (50.0, 2000.0),
            (0.1, 5.0),
            (0.3, 1.0),
            (0.1, 15.0),
            (0.0, 0.9),
            (0.01, 0.6),
            (0.001, 0.3),
            (0.0, 10.0),
            (0.0, 500.0),
            (0.01, 500.0),
            (0.2, 4.0),
            (0.0, 1000.0),
            (-2.0, 2.0),
            (0.6, 1.5),
            (1.0, 7.0),
            (0.0, 0.1),
            (0.0, 0.2),
            (0.0, 3.0),
        ]
        self.params_: Optional[np.ndarray] = None

    @staticmethod
    def _find_feature_idx(feature_names: Sequence[str], keywords: Sequence[str]) -> Optional[int]:
        lower = [str(col).lower() for col in feature_names]
        for i, col in enumerate(lower):
            if all(kw in col for kw in keywords):
                return i
        for i, col in enumerate(lower):
            if any(kw in col for kw in keywords):
                return i
        return None

    @classmethod
    def prepare_forcing(cls, feature_matrix: np.ndarray, feature_names: Sequence[str]):
        x = np.asarray(feature_matrix, dtype=float)
        p_idx = cls._find_feature_idx(feature_names, ["prcp"])
        tmax_idx = cls._find_feature_idx(feature_names, ["tmax"])
        tmin_idx = cls._find_feature_idx(feature_names, ["tmin"])
        dayl_idx = cls._find_feature_idx(feature_names, ["dayl"])
        srad_idx = cls._find_feature_idx(feature_names, ["srad"])

        if p_idx is None:
            raise ValueError('HBV requires a precipitation column containing "prcp".')

        precip = np.clip(x[:, p_idx], 0.0, None)
        if tmax_idx is not None and tmin_idx is not None:
            tmax = x[:, tmax_idx]
            tmin = x[:, tmin_idx]
            temp = (tmax + tmin) / 2.0
            delta_t = np.clip(tmax - tmin, 0.0, None)
        else:
            temp = np.zeros(x.shape[0], dtype=float)
            delta_t = np.full(x.shape[0], 5.0, dtype=float)

        dayl_factor = np.clip(x[:, dayl_idx] / 86400.0, 0.0, 1.0) if dayl_idx is not None else np.ones(x.shape[0], dtype=float)
        if srad_idx is not None:
            rad_mj_day = np.clip(x[:, srad_idx], 0.0, None) * 0.0864
            pet = 0.0023 * rad_mj_day * np.maximum(temp + 17.8, 0.0) * np.sqrt(delta_t + 1e-8)
            pet *= dayl_factor
        else:
            pet = np.clip((temp + 5.0) * 0.05 * np.sqrt(delta_t + 1e-8) * dayl_factor, 0.0, None)
        return precip, temp, np.nan_to_num(pet, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _triangular_weights(maxbas):
        maxbas_int = max(1, int(round(maxbas)))
        weights = np.zeros(maxbas_int, dtype=float)
        half = maxbas / 2.0
        for i in range(maxbas_int):
            t = i + 0.5
            weights[i] = t / half if t <= half else (maxbas - t) / half
        total = weights.sum()
        return weights / total if total > 0 else np.array([1.0], dtype=float)

    @staticmethod
    def _simulate(precip, temp, pet, params, warmup_cycles=1):
        (
            fc, beta, lp, cfmax, k0, k1, k2, perc, uzl, q_scale, p_scale,
            q_base, tt, sfcf, maxbas, cfr, cwh, tt_diff,
        ) = params
        n = precip.shape[0]

        def step(p_t, temp_t, pet_t, snow, liquid_water, soil, upper, lower):
            p_t = max(p_t, 0.0)
            pet_t = max(pet_t, 0.0)
            tt_low = tt - tt_diff
            tt_high = tt + tt_diff
            if tt_diff > 1e-6 and tt_low < temp_t < tt_high:
                rain_frac = (temp_t - tt_low) / (tt_high - tt_low)
            elif temp_t <= tt_low:
                rain_frac = 0.0
            else:
                rain_frac = 1.0

            rain = p_t * rain_frac * p_scale
            snowfall = p_t * (1.0 - rain_frac) * sfcf * p_scale
            snow += snowfall

            if temp_t > tt:
                melt = min(cfmax * (temp_t - tt), snow)
                snow -= melt
                liquid_water += melt
            else:
                refreeze = min(cfr * cfmax * (tt - temp_t), liquid_water)
                liquid_water -= refreeze
                snow += refreeze

            max_liq = cwh * snow
            excess = max(liquid_water - max_liq, 0.0)
            liquid_water = min(liquid_water, max_liq)
            water_input = rain + excess

            sr = min(max(soil / fc, 0.0), 1.0)
            recharge = water_input * (sr ** beta)
            soil += water_input - recharge
            lim_lp = lp * fc
            evap_factor = min(soil / lim_lp, 1.0) if lim_lp > 1e-6 else 1.0
            soil = min(max(soil - pet_t * evap_factor, 0.0), fc)

            upper += recharge
            q0 = k0 * (upper - uzl) if upper > uzl else 0.0
            q1 = k1 * upper
            out = min(q0 + q1, upper)
            pf = min(perc, upper - out)
            upper -= out + pf
            lower += pf
            q2 = k2 * lower
            lower -= q2
            q_t = (out + q2) * q_scale + q_base
            return q_t, snow, liquid_water, soil, upper, lower

        snow = 0.0
        liquid_water = 0.0
        soil = 0.5 * fc
        upper = 10.0
        lower = 10.0

        warmup_len = min(365, n)
        for _ in range(warmup_cycles):
            for i in range(warmup_len):
                _, snow, liquid_water, soil, upper, lower = step(
                    float(precip[i]), float(temp[i]), float(pet[i]), snow, liquid_water, soil, upper, lower
                )

        q_raw = np.zeros(n, dtype=float)
        for i in range(n):
            q_raw[i], snow, liquid_water, soil, upper, lower = step(
                float(precip[i]), float(temp[i]), float(pet[i]), snow, liquid_water, soil, upper, lower
            )

        weights = HBVBenchmark._triangular_weights(maxbas)
        return np.convolve(q_raw, weights, mode="full")[:n] if weights.size > 1 else q_raw

    def _objective(self, params, precip, temp, pet, q_obs):
        q_sim = self._simulate(precip, temp, pet, params, warmup_cycles=0)
        start = min(self.warmup, q_obs.size - 1)
        sim_eval = q_sim[start:]
        obs_eval = q_obs[start:]
        denom = np.sum((obs_eval - np.mean(obs_eval)) ** 2)
        if sim_eval.size < 2 or denom <= 1e-8:
            return 1e6
        nse = 1.0 - np.sum((obs_eval - sim_eval) ** 2) / denom
        return 1.0 - max(nse, -20.0)

    def fit(self, precip, temp, pet, q_obs):
        result = differential_evolution(
            self._objective,
            bounds=self.bounds,
            args=(np.asarray(precip), np.asarray(temp), np.asarray(pet), np.asarray(q_obs)),
            strategy="best1bin",
            maxiter=self.maxiter,
            popsize=10,
            tol=5e-3,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            disp=False,
        )
        self.params_ = result.x
        return self.params_

    def fit_predict(self, train_forcing, val_forcing, test_forcing, train_runoff, val_runoff):
        p_tr, t_tr, pet_tr = train_forcing
        p_val, t_val, pet_val = val_forcing
        p_te, t_te, pet_te = test_forcing
        q_train = np.asarray(train_runoff, dtype=float).reshape(-1)
        q_val = np.asarray(val_runoff, dtype=float).reshape(-1)

        p_calib = np.concatenate([p_tr, p_val], axis=0)
        t_calib = np.concatenate([t_tr, t_val], axis=0)
        pet_calib = np.concatenate([pet_tr, pet_val], axis=0)
        q_calib = np.concatenate([q_train, q_val], axis=0)

        fit_start = time.perf_counter()
        params = self.fit(p_calib, t_calib, pet_calib, q_calib)
        fit_time = time.perf_counter() - fit_start

        infer_start = time.perf_counter()
        pred_train = self._simulate(p_tr, t_tr, pet_tr, params, warmup_cycles=1)
        pred_train_val = self._simulate(
            np.concatenate([p_tr, p_val], axis=0),
            np.concatenate([t_tr, t_val], axis=0),
            np.concatenate([pet_tr, pet_val], axis=0),
            params,
            warmup_cycles=1,
        )
        pred_val = pred_train_val[q_train.size : q_train.size + q_val.size]
        pred_all = self._simulate(
            np.concatenate([p_tr, p_val, p_te], axis=0),
            np.concatenate([t_tr, t_val, t_te], axis=0),
            np.concatenate([pet_tr, pet_val, pet_te], axis=0),
            params,
            warmup_cycles=1,
        )
        pred_test = pred_all[q_train.size + q_val.size :]
        infer_time = time.perf_counter() - infer_start

        return pred_train, pred_val, pred_test, {
            "method": "HBV-Enhanced",
            "params": params.tolist(),
            "fit_time_s": float(fit_time),
            "inference_time_s": float(infer_time),
        }
