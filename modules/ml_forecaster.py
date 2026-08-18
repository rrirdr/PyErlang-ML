"""
ml_forecaster.py
=================
Model zoo (Prophet, XGBoost, SARIMAX, Holt-Winters, weighted Ensemble) with
expanding-window (walk-forward) backtesting for contact-center volume
forecasting at Long-Term (monthly/weekly), Short-Term (daily), and
Interval-Level (30-min) horizons.

TIME-SERIES INTEGRITY MANDATE:
All cross-validation in this module uses EXPANDING WINDOW / WALK-FORWARD
splits. Random K-Fold CV is never used for time-series data because it
leaks future information into the training set (look-ahead bias). Each
fold trains only on data strictly before the fold's validation window.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Optional heavy dependencies are imported lazily / defensively so the rest
# of the app keeps working even if one library is unavailable in the env.
# --------------------------------------------------------------------------- #
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


# --------------------------------------------------------------------------- #
# Metrics (WAPE, MAPE, RMSE, MAE) -- mandated reporting set
# --------------------------------------------------------------------------- #

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error = sum|e| / sum|y_true|. Robust to zeros."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring true-zero intervals to avoid div/0."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "WAPE": wape(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Feature engineering (used by XGBoost)
# --------------------------------------------------------------------------- #

def build_lag_features(
    series: pd.Series,
    lags: tuple[int, ...] = (1, 7, 48),
    rolling_windows: tuple[int, ...] = (7, 14),
) -> pd.DataFrame:
    """
    Build lag, rolling-statistic, and temporal-encoding features for a
    regularly-spaced series indexed by timestamp. Rows with insufficient
    history (NaN lags) are left in; the caller drops them before fitting.
    """
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w, min_periods=1).std()

    idx = df.index
    df["dow"] = idx.dayofweek
    df["hour"] = idx.hour if hasattr(idx, "hour") else 0
    df["day"] = idx.day
    df["month"] = idx.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # Cyclical encodings so the model understands 23:00 is close to 00:00
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


# --------------------------------------------------------------------------- #
# Individual model wrappers -- each exposes fit(train) -> predict(n_periods)
# --------------------------------------------------------------------------- #

@dataclass
class ForecastOutput:
    dates: pd.DatetimeIndex
    values: np.ndarray
    model_name: str


class BaseForecaster:
    name: str = "base"

    def fit(self, series: pd.Series, freq: str) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, n_periods: int) -> np.ndarray:
        raise NotImplementedError


class ProphetForecaster(BaseForecaster):
    name = "Prophet"

    def __init__(self, yearly=True, weekly=True, daily=True):
        self.yearly, self.weekly, self.daily = yearly, weekly, daily
        self._model = None
        self._freq = None

    def fit(self, series: pd.Series, freq: str):
        if not _HAS_PROPHET:
            raise RuntimeError("Prophet is not installed in this environment.")
        self._freq = freq
        df = pd.DataFrame({"ds": series.index, "y": series.values})
        self._model = Prophet(
            yearly_seasonality=self.yearly,
            weekly_seasonality=self.weekly,
            daily_seasonality=self.daily,
            interval_width=0.8,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(df)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        future = self._model.make_future_dataframe(periods=n_periods, freq=self._freq, include_history=False)
        fcst = self._model.predict(future)
        return np.clip(fcst["yhat"].values, a_min=0, a_max=None)


class XGBoostForecaster(BaseForecaster):
    name = "XGBoost"

    def __init__(self, lags=(1, 7, 48), rolling_windows=(7, 14), n_estimators=300, max_depth=5, lr=0.05):
        self.lags, self.rolling_windows = lags, rolling_windows
        self.n_estimators, self.max_depth, self.lr = n_estimators, max_depth, lr
        self._model = None
        self._history: Optional[pd.Series] = None
        self._freq = None

    def fit(self, series: pd.Series, freq: str):
        if not _HAS_XGBOOST:
            raise RuntimeError("XGBoost is not installed in this environment.")
        self._freq = freq
        self._history = series.copy()
        feats = build_lag_features(series, self.lags, self.rolling_windows).dropna()
        X = feats.drop(columns=["y"])
        y = feats["y"]
        self._feature_cols = X.columns.tolist()
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.lr, subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=42, verbosity=0,
        )
        self._model.fit(X, y)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        # Recursive multi-step forecasting: append each prediction to the
        # working history so subsequent lag features can reference it.
        working = self._history.copy()
        future_idx = pd.date_range(
            start=working.index[-1] + (working.index[-1] - working.index[-2]),
            periods=n_periods, freq=self._freq,
        )
        preds = []
        for ts in future_idx:
            extended = pd.concat([working, pd.Series([np.nan], index=[ts])])
            feats = build_lag_features(extended, self.lags, self.rolling_windows)
            row = feats.iloc[[-1]][self._feature_cols]
            row = row.fillna(0.0)
            pred = float(self._model.predict(row)[0])
            pred = max(pred, 0.0)
            preds.append(pred)
            working.loc[ts] = pred
        return np.array(preds)


class SARIMAXForecaster(BaseForecaster):
    name = "SARIMAX"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order, self.seasonal_order = order, seasonal_order
        self._model_fit = None
        self._freq = None

    def fit(self, series: pd.Series, freq: str):
        if not _HAS_STATSMODELS:
            raise RuntimeError("statsmodels is not installed in this environment.")
        self._freq = freq
        try:
            model = SARIMAX(
                series.values, order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            self._model_fit = model.fit(disp=False)
        except Exception:
            # Fall back to a simpler non-seasonal order if the seasonal fit fails
            # to converge (common on short/irregular histories).
            model = SARIMAX(series.values, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            self._model_fit = model.fit(disp=False)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        fc = self._model_fit.forecast(steps=n_periods)
        return np.clip(np.asarray(fc), a_min=0, a_max=None)


class HoltWintersForecaster(BaseForecaster):
    name = "Holt-Winters"

    def __init__(self, seasonal_periods: int = 7, trend="add", seasonal="add"):
        self.seasonal_periods, self.trend, self.seasonal = seasonal_periods, trend, seasonal
        self._model_fit = None

    def fit(self, series: pd.Series, freq: str):
        if not _HAS_STATSMODELS:
            raise RuntimeError("statsmodels is not installed in this environment.")
        vals = series.values.astype(float)
        vals = np.where(vals <= 0, 1e-3, vals)  # HW multiplicative needs strictly positive values
        sp = self.seasonal_periods if len(vals) >= 2 * self.seasonal_periods else None
        seasonal = self.seasonal if sp else None
        try:
            model = ExponentialSmoothing(
                vals, trend=self.trend, seasonal=seasonal, seasonal_periods=sp,
                initialization_method="estimated",
            )
            self._model_fit = model.fit()
        except Exception:
            model = ExponentialSmoothing(vals, trend="add", seasonal=None, initialization_method="estimated")
            self._model_fit = model.fit()
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        fc = self._model_fit.forecast(n_periods)
        return np.clip(np.asarray(fc), a_min=0, a_max=None)


class EnsembleForecaster(BaseForecaster):
    """Performance-weighted average of a set of already-fitted base models."""
    name = "Ensemble"

    def __init__(self, models: list[BaseForecaster], weights: list[float]):
        assert len(models) == len(weights) and len(models) > 0
        self.models = models
        total = sum(weights)
        self.weights = [w / total for w in weights] if total > 0 else [1 / len(weights)] * len(weights)

    def predict(self, n_periods: int) -> np.ndarray:
        preds = np.array([m.predict(n_periods) for m in self.models])
        return np.clip(np.average(preds, axis=0, weights=self.weights), a_min=0, a_max=None)


MODEL_REGISTRY: dict[str, Callable[[], BaseForecaster]] = {
    "Prophet": lambda: ProphetForecaster(),
    "XGBoost": lambda: XGBoostForecaster(),
    "SARIMAX": lambda: SARIMAXForecaster(),
    "Holt-Winters": lambda: HoltWintersForecaster(),
}


def available_models() -> list[str]:
    """Models actually usable in this runtime (skips missing optional deps)."""
    names = []
    if _HAS_PROPHET:
        names.append("Prophet")
    if _HAS_XGBOOST:
        names.append("XGBoost")
    if _HAS_STATSMODELS:
        names.extend(["SARIMAX", "Holt-Winters"])
    return names


# --------------------------------------------------------------------------- #
# Expanding-window (walk-forward) backtesting
# --------------------------------------------------------------------------- #

@dataclass
class BacktestFoldResult:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict


@dataclass
class ModelBacktestSummary:
    model_name: str
    folds: list[BacktestFoldResult] = field(default_factory=list)
    avg_metrics: dict = field(default_factory=dict)
    fit_error: Optional[str] = None


def expanding_window_backtest(
    series: pd.Series,
    freq: str,
    model_names: list[str],
    horizon: int,
    n_folds: int = 3,
    min_train_size: Optional[int] = None,
) -> list[ModelBacktestSummary]:
    """
    Walk-forward validation: for fold i, train on all data up to a cutoff,
    predict the next `horizon` periods, then advance the cutoff forward by
    `horizon` periods for the next fold. The training window only ever
    EXPANDS (never shrinks, never samples randomly) -- this eliminates
    look-ahead bias inherent to K-Fold CV on time-series.

    Returns one ModelBacktestSummary per requested model, each containing
    per-fold metrics plus the fold-averaged metrics for leaderboard ranking.
    """
    n = len(series)
    min_train_size = min_train_size or max(2 * horizon, 14)

    # Determine feasible fold cutoffs walking backward from the end of history
    max_possible_folds = max((n - min_train_size) // horizon, 0)
    n_folds = max(min(n_folds, max_possible_folds), 1)

    cutoffs = []
    for i in range(n_folds):
        test_end_idx = n - (n_folds - 1 - i) * horizon
        test_start_idx = test_end_idx - horizon
        train_end_idx = test_start_idx
        if train_end_idx < min_train_size:
            continue
        cutoffs.append((train_end_idx, test_start_idx, test_end_idx))

    summaries = []
    for model_name in model_names:
        summary = ModelBacktestSummary(model_name=model_name)
        try:
            for fold_i, (train_end_idx, test_start_idx, test_end_idx) in enumerate(cutoffs):
                train = series.iloc[:train_end_idx]
                test = series.iloc[test_start_idx:test_end_idx]
                if len(test) == 0 or len(train) < min_train_size:
                    continue

                model = MODEL_REGISTRY[model_name]()
                model.fit(train, freq)
                y_pred = model.predict(len(test))
                y_true = test.values

                m = compute_all_metrics(y_true, y_pred)
                summary.folds.append(BacktestFoldResult(
                    fold=fold_i, train_end=train.index[-1], test_start=test.index[0],
                    test_end=test.index[-1], y_true=y_true, y_pred=y_pred, metrics=m,
                ))

            if summary.folds:
                keys = summary.folds[0].metrics.keys()
                summary.avg_metrics = {
                    k: float(np.nanmean([f.metrics[k] for f in summary.folds])) for k in keys
                }
            else:
                summary.fit_error = "Not enough history for any valid fold at this horizon."
        except Exception as e:
            summary.fit_error = f"{type(e).__name__}: {e}"

        summaries.append(summary)

    return summaries


def fit_final_models_and_forecast(
    series: pd.Series,
    freq: str,
    model_names: list[str],
    horizon: int,
    backtest_summaries: Optional[list[ModelBacktestSummary]] = None,
) -> dict[str, ForecastOutput]:
    """
    Fit each requested model on the FULL available history and produce the
    forward-looking forecast of length `horizon`. If backtest_summaries are
    supplied, also builds a performance-weighted Ensemble forecast (weight
    inversely proportional to backtested WAPE).
    """
    results: dict[str, ForecastOutput] = {}
    fitted_models: dict[str, BaseForecaster] = {}

    last_ts = series.index[-1]
    step = series.index[-1] - series.index[-2] if len(series) > 1 else pd.Timedelta(days=1)
    future_dates = pd.date_range(start=last_ts + step, periods=horizon, freq=freq)

    for name in model_names:
        try:
            model = MODEL_REGISTRY[name]()
            model.fit(series, freq)
            preds = model.predict(horizon)
            results[name] = ForecastOutput(dates=future_dates, values=preds, model_name=name)
            fitted_models[name] = model
        except Exception:
            continue  # skip models that fail to fit; leaderboard will show why

    if backtest_summaries and len(fitted_models) > 1:
        weights = []
        ordered_models = []
        for s in backtest_summaries:
            if s.model_name in fitted_models and s.avg_metrics.get("WAPE") not in (None, float("nan")):
                w = 1.0 / max(s.avg_metrics["WAPE"], 1e-3)
                weights.append(w)
                ordered_models.append(fitted_models[s.model_name])
        if len(ordered_models) > 1:
            ens = EnsembleForecaster(ordered_models, weights)
            preds = ens.predict(horizon)
            results["Ensemble"] = ForecastOutput(dates=future_dates, values=preds, model_name="Ensemble")

    return results


# --------------------------------------------------------------------------- #
# Intraday profile splitting (Day-of-Week x Interval matrix)
# --------------------------------------------------------------------------- #

def build_intraday_profile(interval_df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """
    Build a Day-of-Week x Time-of-Day matrix of the historical share of
    daily volume that falls in each interval slot. Used to split a
    daily-level forecast down into 30-min interval forecasts.
    """
    df = interval_df.copy()
    df["date"] = df["timestamp"].dt.date
    df["dow"] = df["timestamp"].dt.dayofweek
    df["hm"] = df["timestamp"].dt.strftime("%H:%M")

    daily_totals = df.groupby("date")["volume"].transform("sum").replace(0, np.nan)
    df["share"] = df["volume"] / daily_totals

    profile = df.groupby(["dow", "hm"])["share"].mean().reset_index()
    # Renormalize each day-of-week's shares to sum to 1.0 exactly
    profile["share"] = profile.groupby("dow")["share"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 1.0 / len(s)
    )
    return profile


def split_daily_forecast_to_intervals(
    daily_forecast: ForecastOutput, profile: pd.DataFrame, interval_minutes: int
) -> pd.DataFrame:
    """Explode a daily-level forecast into 30-min (or other grain) intervals using the intraday profile."""
    rows = []
    n_slots_per_day = int(24 * 60 / interval_minutes)
    slot_times = [pd.Timestamp("2000-01-01") + pd.Timedelta(minutes=interval_minutes * i) for i in range(n_slots_per_day)]
    slot_hms = [t.strftime("%H:%M") for t in slot_times]

    for date, total in zip(daily_forecast.dates, daily_forecast.values):
        dow = pd.Timestamp(date).dayofweek
        day_profile = profile[profile["dow"] == dow].set_index("hm")["share"]
        for hm in slot_hms:
            share = day_profile.get(hm, 1.0 / n_slots_per_day)
            ts = pd.Timestamp(date) + (pd.Timestamp("2000-01-01") + pd.Timedelta(minutes=0) - pd.Timestamp("2000-01-01"))
            slot_ts = pd.Timestamp(pd.Timestamp(date).date()) + pd.Timedelta(
                hours=int(hm[:2]), minutes=int(hm[3:5])
            )
            rows.append({"timestamp": slot_ts, "volume": total * share})

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
