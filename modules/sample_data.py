"""
sample_data.py
===============
Generates a realistic synthetic contact-center interval dataset so users can
explore PyErlang-ML end-to-end without uploading their own file, plus a
minimal downloadable CSV template that documents the exact schema expected
for real uploads.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def generate_sample_dataset(
    n_days: int = 120,
    interval_minutes: int = 30,
    base_volume: float = 60.0,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Build a synthetic 30-min interval call-volume dataset with:
      - Intraday U/peak shape (low overnight, peak mid-morning & early afternoon)
      - Weekday vs weekend volume differential
      - A gentle upward trend over the period (so forecasting models have
        something meaningful to project forward)
      - Random noise, a handful of injected volume spikes (outliers), and a
        handful of dropped rows (missing intervals) so the Data Hygiene tab
        has real work to do when a user explores it.
      - A variable AHT column with small random variation.

    Returns a dataframe with columns: timestamp, volume, aht -- ready to be
    dropped directly into the same pipeline as an uploaded file.
    """
    rng = np.random.default_rng(seed)

    n_slots_per_day = int(24 * 60 / interval_minutes)
    total_slots = n_days * n_slots_per_day

    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days)
    timestamps = pd.date_range(start=start, periods=total_slots, freq=f"{interval_minutes}min")

    hour_of_day = np.array(timestamps.hour) + np.array(timestamps.minute) / 60.0
    day_of_week = np.array(timestamps.dayofweek)  # 0=Mon ... 6=Sun
    day_index = np.arange(total_slots) // n_slots_per_day

    # Intraday shape: two humps (late morning + early afternoon), near-zero overnight
    intraday = (
        np.exp(-((hour_of_day - 10.5) ** 2) / (2 * 2.2 ** 2)) * 1.0
        + np.exp(-((hour_of_day - 14.5) ** 2) / (2 * 2.5 ** 2)) * 0.8
    )
    intraday = intraday / intraday.max()

    # Weekday vs weekend multiplier
    weekday_mult = np.where(day_of_week < 5, 1.0, 0.35)

    # Gentle upward trend across the whole period (~+15% end vs start)
    trend = 1.0 + 0.15 * (day_index / max(day_index.max(), 1))

    # Slight day-of-week texture (e.g. Mondays busier, Fridays a bit lighter)
    dow_texture = np.take([1.08, 1.02, 1.0, 1.0, 0.95, 0.35, 0.30], day_of_week)

    volume = base_volume * intraday * weekday_mult * dow_texture * trend
    noise = rng.normal(0, base_volume * 0.06, total_slots)
    volume = np.clip(volume + noise, 0, None)

    # Inject a handful of realistic outlier spikes (e.g. marketing campaign, outage backlog)
    n_spikes = max(int(total_slots * 0.004), 3)
    spike_idx = rng.choice(total_slots, size=n_spikes, replace=False)
    volume[spike_idx] = volume[spike_idx] * rng.uniform(2.5, 4.5, size=n_spikes)

    # AHT: baseline ~300s with mild random variation and slightly longer handle
    # times during peak hours (fatigue / complexity effect)
    aht = 300 + rng.normal(0, 18, total_slots) + (intraday * 25)
    aht = np.clip(aht, 90, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "volume": np.round(volume, 0),
        "aht": np.round(aht, 0),
    })

    # Drop a handful of rows to simulate real-world missing intervals
    n_missing = max(int(total_slots * 0.01), 5)
    drop_idx = rng.choice(df.index, size=n_missing, replace=False)
    df = df.drop(index=drop_idx).reset_index(drop=True)

    return df


def sample_dataset_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize the sample dataset to CSV bytes for an optional direct download."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def generate_template_csv_bytes(n_example_rows: int = 8) -> bytes:
    """
    Build a minimal, well-formed CSV template that documents the exact
    schema PyErlang-ML expects, pre-filled with a few example rows so users
    can see valid formatting at a glance before dropping in their own data.
    """
    start = pd.Timestamp("2026-01-05 00:00:00")  # a Monday, for clarity
    rows = []
    for i in range(n_example_rows):
        ts = start + pd.Timedelta(minutes=30 * i)
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "volume": [42, 38, 55, 61, 0, 47, 50, 33][i % 8],
            "aht": [312, 298, 305, 290, 300, 315, 288, 300][i % 8],
        })
    template_df = pd.DataFrame(rows)

    buf = io.StringIO()
    buf.write("# PyErlang-ML data template\n")
    buf.write("# Required columns: timestamp, volume | Optional: aht (seconds)\n")
    buf.write("# - timestamp: any parseable date/time; keep a consistent, regular interval (e.g. 15 or 30 min)\n")
    buf.write("# - volume: contacts offered in that interval (0 is fine for overnight/closed intervals)\n")
    buf.write("# - aht: average handle time in SECONDS; omit this column entirely if you don't track it per-interval\n")
    buf.write("# Delete these comment lines and the example rows below before uploading your own data.\n")
    template_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
