"""
capacity_planner.py
====================
Multi-layered shrinkage solver, occupancy capping, and net/gross FTE
requirement generator. Bridges raw Erlang C/A staffing requirements to a
real-world staffing plan that accounts for occupancy ceilings and
compound (planned x unplanned) shrinkage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import erlang_engine as ee


@dataclass
class IntervalPlan:
    timestamp: pd.Timestamp
    forecast_volume: float
    aht_seconds: float
    raw_agents: int              # Erlang-optimal agents to hit SL target
    occupancy_capped_agents: int  # after applying max occupancy ceiling
    net_fte: float                # operational floor staff needed
    gross_fte: float              # net_fte grossed up for shrinkage
    predicted_occupancy: float
    predicted_sl: float
    predicted_asa: float
    predicted_abandon: float
    engine_used: str              # "erlang_c" or "erlang_a"


def compound_shrinkage_factor(planned_shrinkage: float, unplanned_shrinkage: float) -> float:
    """
    Returns the multiplier applied to Net FTE to reach Gross FTE:
        Gross FTE = Net FTE / [(1 - planned) * (1 - unplanned)]

    Both shrinkage inputs are fractions in [0, 1). Defensive clamping
    prevents divide-by-zero / negative-FTE outputs at the 100% boundary.
    """
    p = min(max(planned_shrinkage, 0.0), 0.98)
    u = min(max(unplanned_shrinkage, 0.0), 0.98)
    denom = (1 - p) * (1 - u)
    denom = max(denom, 0.02)  # floor to avoid explosive/undefined gross FTE
    return 1.0 / denom


def apply_occupancy_cap(raw_agents: int, traffic_intensity_erlangs: float, max_occupancy: float) -> int:
    """
    Occupancy = A / c. If the Erlang-optimal agent count would push agents
    above the max allowed occupancy (e.g. 85%), add agents until occupancy
    is at or below the cap. Never reduces agents below what SL requires.
    """
    if raw_agents <= 0 or traffic_intensity_erlangs <= 0:
        return raw_agents
    max_occupancy = min(max(max_occupancy, 0.01), 0.999)
    min_agents_for_occ_cap = math.ceil(traffic_intensity_erlangs / max_occupancy)
    return max(raw_agents, min_agents_for_occ_cap)


def plan_interval(
    timestamp: pd.Timestamp,
    forecast_volume: float,
    aht_seconds: float,
    interval_seconds: int,
    target_sl: float,
    target_answer_seconds: int,
    max_occupancy: float,
    planned_shrinkage: float,
    unplanned_shrinkage: float,
    use_erlang_a: bool = False,
    patience_seconds: Optional[float] = None,
    max_agents_search: int = 1500,
) -> IntervalPlan:
    """
    Full pipeline for a single interval: forecast volume -> raw Erlang
    staffing -> occupancy-capped staffing -> net/gross FTE -> predicted
    performance at the final staffing level.
    """
    # Defensive: zero-volume or zero-AHT intervals require zero staff and
    # produce perfect (trivial) performance rather than crashing the solver.
    if forecast_volume <= 0 or aht_seconds <= 0:
        return IntervalPlan(
            timestamp=timestamp, forecast_volume=max(forecast_volume, 0.0), aht_seconds=aht_seconds,
            raw_agents=0, occupancy_capped_agents=0, net_fte=0.0, gross_fte=0.0,
            predicted_occupancy=0.0, predicted_sl=1.0, predicted_asa=0.0, predicted_abandon=0.0,
            engine_used="erlang_a" if use_erlang_a else "erlang_c",
        )

    A = (forecast_volume * aht_seconds) / interval_seconds

    if use_erlang_a and patience_seconds:
        raw_agents = ee.required_agents_erlang_a(
            forecast_volume, aht_seconds, interval_seconds, target_sl,
            target_answer_seconds, patience_seconds, max_agents=max_agents_search,
        )
        engine = "erlang_a"
    else:
        raw_agents = ee.required_agents_erlang_c(
            forecast_volume, aht_seconds, interval_seconds, target_sl,
            target_answer_seconds, max_agents=max_agents_search,
        )
        engine = "erlang_c"

    capped_agents = apply_occupancy_cap(raw_agents, A, max_occupancy)

    net_fte = float(capped_agents)
    gross_fte = net_fte * compound_shrinkage_factor(planned_shrinkage, unplanned_shrinkage)

    # Predicted performance at the FINAL (occupancy-capped) staffing level
    if engine == "erlang_a":
        perf = ee.erlang_a_metrics(
            forecast_volume, aht_seconds, interval_seconds, capped_agents,
            target_answer_seconds, patience_seconds,
        )
        predicted_abandon = perf.prob_abandon
    else:
        perf = ee.erlang_c_metrics(
            forecast_volume, aht_seconds, interval_seconds, capped_agents, target_answer_seconds,
        )
        predicted_abandon = 0.0

    return IntervalPlan(
        timestamp=timestamp, forecast_volume=forecast_volume, aht_seconds=aht_seconds,
        raw_agents=raw_agents, occupancy_capped_agents=capped_agents,
        net_fte=net_fte, gross_fte=gross_fte,
        predicted_occupancy=perf.occupancy if np.isfinite(perf.occupancy) else 1.0,
        predicted_sl=perf.service_level, predicted_asa=perf.asa_seconds if np.isfinite(perf.asa_seconds) else 0.0,
        predicted_abandon=predicted_abandon, engine_used=engine,
    )


def build_staffing_plan(
    interval_forecast_df: pd.DataFrame,   # columns: timestamp, volume, [aht]
    interval_seconds: int,
    target_sl: float,
    target_answer_seconds: int,
    max_occupancy: float,
    planned_shrinkage: float,
    unplanned_shrinkage: float,
    default_aht_seconds: float = 300.0,
    use_erlang_a: bool = False,
    patience_seconds: Optional[float] = None,
) -> pd.DataFrame:
    """
    Vectorized-orchestration wrapper: runs plan_interval() across every row
    of a forecast dataframe and returns a tidy staffing-plan dataframe ready
    for visualization and export.
    """
    aht_col = interval_forecast_df["aht"] if "aht" in interval_forecast_df.columns else pd.Series(
        default_aht_seconds, index=interval_forecast_df.index
    )

    records = []
    for i, row in interval_forecast_df.reset_index(drop=True).iterrows():
        aht = aht_col.iloc[i] if aht_col.iloc[i] and aht_col.iloc[i] > 0 else default_aht_seconds
        plan = plan_interval(
            timestamp=row["timestamp"], forecast_volume=max(row["volume"], 0.0), aht_seconds=aht,
            interval_seconds=interval_seconds, target_sl=target_sl, target_answer_seconds=target_answer_seconds,
            max_occupancy=max_occupancy, planned_shrinkage=planned_shrinkage, unplanned_shrinkage=unplanned_shrinkage,
            use_erlang_a=use_erlang_a, patience_seconds=patience_seconds,
        )
        records.append(plan.__dict__)

    return pd.DataFrame(records)


def apply_whatif_scenario(
    base_forecast_df: pd.DataFrame,
    volume_pct_change: float = 0.0,
    aht_pct_change: float = 0.0,
    planned_shrinkage_delta: float = 0.0,
    unplanned_shrinkage_delta: float = 0.0,
    sl_target_override: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply what-if sandbox deltas to a base interval forecast without
    mutating the original dataframe. Returns a scenario-adjusted copy.
    """
    scenario = base_forecast_df.copy()
    scenario["volume"] = (scenario["volume"] * (1 + volume_pct_change)).clip(lower=0.0)
    if "aht" in scenario.columns:
        scenario["aht"] = (scenario["aht"] * (1 + aht_pct_change)).clip(lower=1.0)
    return scenario
