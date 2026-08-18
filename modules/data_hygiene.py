"""
data_hygiene.py
================
Time-series parsing, interval-grain detection, missing-slot imputation,
and outlier smoothing (IQR or Z-score) for contact-center interval data.

Expected raw schema (case-insensitive column matching, flexible order):
    timestamp   : datetime-like
    volume      : numeric, contacts offered
    aht         : numeric, seconds (optional -- a default is used if absent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "volume"}
OPTIONAL_COLUMNS = {"aht"}


@dataclass
class HygieneReport:
    """Summary of everything the hygiene pipeline did to the raw data."""
    detected_interval_minutes: int = 30
    rows_in: int = 0
    rows_out: int = 0
    missing_slots_found: int = 0
    missing_slots_imputed: int = 0
    outliers_flagged: int = 0
    outliers_smoothed: int = 0
    zero_volume_intervals: int = 0
    zero_aht_intervals: int = 0
    total_volume: float = 0.0
    avg_daily_volume: float = 0.0
    peak_interval_volume: float = 0.0
    peak_interval_timestamp: Optional[pd.Timestamp] = None
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Schema handling
# --------------------------------------------------------------------------- #

def normalize_schema(df: pd.DataFrame, default_aht: float = 300.0) -> pd.DataFrame:
    """
    Case-insensitively map user columns onto the canonical schema:
    timestamp, volume, aht. Raises ValueError if required columns are missing.
    """
    col_map = {c.lower().strip(): c for c in df.columns}

    ts_col = next((col_map[c] for c in col_map if c in ("timestamp", "date", "datetime", "interval")), None)
    vol_col = next((col_map[c] for c in col_map if c in ("volume", "calls", "contacts", "offered")), None)
    aht_col = next((col_map[c] for c in col_map if c in ("aht", "handletime", "handle_time", "aht_sec")), None)

    if ts_col is None or vol_col is None:
        raise ValueError(
            "Uploaded file must contain a 'timestamp' column and a 'volume' column "
            f"(found columns: {list(df.columns)})."
        )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    out["volume"] = pd.to_numeric(df[vol_col], errors="coerce")

    if aht_col is not None:
        out["aht"] = pd.to_numeric(df[aht_col], errors="coerce")
        out["aht"] = out["aht"].fillna(default_aht)
    else:
        out["aht"] = default_aht

    # Drop rows where timestamp failed to parse entirely (unrecoverable)
    n_bad_ts = out["timestamp"].isna().sum()
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Defensive: volume must be non-negative; negative entries are data errors -> clip to 0
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    out["aht"] = out["aht"].fillna(default_aht).clip(lower=0.0)

    return out


# --------------------------------------------------------------------------- #
# Interval detection
# --------------------------------------------------------------------------- #

def detect_interval_minutes(df: pd.DataFrame) -> int:
    """
    Infer the dominant interval grain (e.g. 15 or 30 minutes) from the most
    common gap between consecutive timestamps. Falls back to 30 if the data
    is too sparse or irregular to infer confidently.
    """
    if len(df) < 3:
        return 30

    deltas = df["timestamp"].diff().dropna().dt.total_seconds() / 60.0
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 30

    mode_val = deltas.mode()
    if mode_val.empty:
        return 30

    inferred = int(round(mode_val.iloc[0]))
    # Snap to a sane standard grain if close
    for standard in (5, 10, 15, 30, 60):
        if abs(inferred - standard) <= 2:
            return standard
    return max(inferred, 1)


# --------------------------------------------------------------------------- #
# Missing-slot imputation (seasonal median: same weekday + time-of-day)
# --------------------------------------------------------------------------- #

def impute_missing_slots(df: pd.DataFrame, interval_minutes: int) -> tuple[pd.DataFrame, int, int]:
    """
    Reindex to a complete, regular timeline at `interval_minutes` grain and
    fill any gaps using the seasonal median volume for the same
    (day-of-week, time-of-day) slot. AHT is filled via the same seasonal
    median, falling back to the global median AHT.

    Returns (filled_df, n_missing_found, n_missing_imputed).
    """
    if df.empty:
        return df, 0, 0

    full_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq=f"{interval_minutes}min",
    )

    indexed = df.set_index("timestamp")
    # Collapse potential duplicate timestamps by summing volume / averaging AHT
    indexed = indexed.groupby(level=0).agg({"volume": "sum", "aht": "mean"})
    reindexed = indexed.reindex(full_index)

    n_missing = int(reindexed["volume"].isna().sum())

    # Build seasonal lookup keyed on (dayofweek, hour, minute) from the
    # non-missing observations for a smarter fill than a flat global median.
    observed = reindexed.dropna(subset=["volume"]).copy()
    observed["dow"] = observed.index.dayofweek
    observed["hm"] = observed.index.strftime("%H:%M")

    seasonal_vol = observed.groupby(["dow", "hm"])["volume"].median()
    seasonal_aht = observed.groupby(["dow", "hm"])["aht"].median()
    global_vol_median = observed["volume"].median() if not observed.empty else 0.0
    global_aht_median = observed["aht"].median() if not observed.empty else 300.0

    reindexed["dow"] = reindexed.index.dayofweek
    reindexed["hm"] = reindexed.index.strftime("%H:%M")

    def _fill_vol(row):
        if not pd.isna(row["volume"]):
            return row["volume"]
        key = (row["dow"], row["hm"])
        return seasonal_vol.get(key, global_vol_median)

    def _fill_aht(row):
        if not pd.isna(row["aht"]):
            return row["aht"]
        key = (row["dow"], row["hm"])
        return seasonal_aht.get(key, global_aht_median)

    reindexed["volume"] = reindexed.apply(_fill_vol, axis=1)
    reindexed["aht"] = reindexed.apply(_fill_aht, axis=1)

    reindexed = reindexed.drop(columns=["dow", "hm"]).reset_index().rename(columns={"index": "timestamp"})
    n_imputed = n_missing  # every missing slot gets a value (never left NaN)

    return reindexed, n_missing, n_imputed


# --------------------------------------------------------------------------- #
# Outlier detection & smoothing
# --------------------------------------------------------------------------- #

def flag_and_smooth_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
) -> tuple[pd.DataFrame, int, int]:
    """
    Detect volume outliers using either IQR or Z-score thresholding
    (computed per day-of-week to respect weekly seasonality), and smooth
    flagged points by replacing them with the seasonal (dow, time-of-day)
    median. Returns (df_with_cleaned_volume, n_flagged, n_smoothed).
    """
    out = df.copy()
    out["is_outlier"] = False
    out["volume_raw"] = out["volume"]

    if out.empty:
        return out, 0, 0

    out["dow"] = out["timestamp"].dt.dayofweek
    out["hm"] = out["timestamp"].dt.strftime("%H:%M")

    def _flag_group(g: pd.DataFrame) -> pd.Series:
        vals = g["volume"]
        if len(vals) < 4:
            return pd.Series(False, index=g.index)
        if method.lower() == "zscore":
            mu, sigma = vals.mean(), vals.std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                return pd.Series(False, index=g.index)
            z = (vals - mu) / sigma
            return z.abs() > z_threshold
        else:  # IQR (default)
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                return pd.Series(False, index=g.index)
            lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
            return (vals < lower) | (vals > upper)

    flags = out.groupby("dow", group_keys=False).apply(_flag_group)
    out["is_outlier"] = flags.reindex(out.index).fillna(False)

    n_flagged = int(out["is_outlier"].sum())

    # Smooth flagged points with seasonal (dow, hm) median computed from
    # NON-outlier points only, so outliers don't contaminate their own fill value.
    clean_only = out[~out["is_outlier"]]
    seasonal_median = clean_only.groupby(["dow", "hm"])["volume"].median()
    global_median = clean_only["volume"].median() if not clean_only.empty else out["volume"].median()

    def _smooth(row):
        if not row["is_outlier"]:
            return row["volume"]
        key = (row["dow"], row["hm"])
        return seasonal_median.get(key, global_median)

    out["volume"] = out.apply(_smooth, axis=1)
    n_smoothed = n_flagged

    out = out.drop(columns=["dow", "hm"])
    return out, n_flagged, n_smoothed


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #

def run_hygiene_pipeline(
    raw_df: pd.DataFrame,
    default_aht: float = 300.0,
    outlier_method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
) -> tuple[pd.DataFrame, HygieneReport]:
    """
    Orchestrates: schema normalization -> interval detection -> missing-slot
    imputation -> outlier flag & smoothing. Returns the cleaned dataframe
    (with both 'volume' [cleaned] and 'volume_raw' [pre-smoothing] columns)
    plus a HygieneReport summarizing everything that was done.
    """
    report = HygieneReport()
    report.rows_in = len(raw_df)

    if raw_df.empty:
        report.notes.append("Input file was empty.")
        return raw_df, report

    normalized = normalize_schema(raw_df, default_aht=default_aht)

    interval_minutes = detect_interval_minutes(normalized)
    report.detected_interval_minutes = interval_minutes

    filled, n_missing, n_imputed = impute_missing_slots(normalized, interval_minutes)
    report.missing_slots_found = n_missing
    report.missing_slots_imputed = n_imputed

    cleaned, n_flagged, n_smoothed = flag_and_smooth_outliers(
        filled, method=outlier_method, iqr_multiplier=iqr_multiplier, z_threshold=z_threshold
    )
    report.outliers_flagged = n_flagged
    report.outliers_smoothed = n_smoothed

    # Defensive edge cases: zero-volume intervals & zero-AHT intervals should
    # never crash downstream Erlang math -- flag them, but leave in the data.
    report.zero_volume_intervals = int((cleaned["volume"] <= 0).sum())
    report.zero_aht_intervals = int((cleaned["aht"] <= 0).sum())
    if report.zero_aht_intervals > 0:
        cleaned.loc[cleaned["aht"] <= 0, "aht"] = default_aht
        report.notes.append(
            f"{report.zero_aht_intervals} interval(s) had 0-second AHT; "
            f"backfilled with default AHT ({default_aht:.0f}s) to keep Erlang math well-defined."
        )

    report.rows_out = len(cleaned)
    report.total_volume = float(cleaned["volume"].sum())

    n_days = max(cleaned["timestamp"].dt.date.nunique(), 1)
    report.avg_daily_volume = report.total_volume / n_days

    if not cleaned.empty:
        peak_idx = cleaned["volume"].idxmax()
        report.peak_interval_volume = float(cleaned.loc[peak_idx, "volume"])
        report.peak_interval_timestamp = cleaned.loc[peak_idx, "timestamp"]

    return cleaned, report
