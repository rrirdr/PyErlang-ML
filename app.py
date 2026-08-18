"""
app.py
======
PyErlang-ML — Enterprise Call Volume Forecasting & Erlang Capacity Engine.
Main Streamlit layout: sidebar global settings + 4-tab workflow.
"""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from modules import data_hygiene as dh
from modules import ml_forecaster as mlf
from modules import capacity_planner as cp
from modules import exporter as exp

st.set_page_config(page_title="PyErlang-ML", layout="wide", page_icon="📞")


# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #

DEFAULT_STATE = {
    "raw_df": None,
    "cleaned_df": None,
    "hygiene_report": None,
    "interval_minutes": 30,
    "forecasts": {},          # {model_name: ForecastOutput}
    "backtest_summaries": [],
    "forecast_horizon_label": None,
    "interval_forecast_df": None,   # final interval-level forecast used for capacity planning
    "staffing_plan_df": None,
    "chosen_forecast_model": None,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --------------------------------------------------------------------------- #
# Sidebar — Global Settings
# --------------------------------------------------------------------------- #

st.sidebar.title("📞 PyErlang-ML")
st.sidebar.caption("Enterprise Call Volume Forecasting & Erlang Capacity Engine")
st.sidebar.divider()

st.sidebar.subheader("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload interval volume file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded_file)
        else:
            raw = pd.read_excel(uploaded_file)
        st.session_state["raw_df"] = raw
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

st.sidebar.subheader("2. WFM Assumptions")
target_sl_pct = st.sidebar.slider("Target Service Level (%)", 50, 99, 80, 1)
target_sl_time = st.sidebar.number_input("Target SL Answer Time (sec)", min_value=1, value=20, step=1)
mta_seconds = st.sidebar.number_input("Mean Time to Abandon — MTA (sec)", min_value=1, value=90, step=5)
max_occupancy_pct = st.sidebar.slider("Max Occupancy Cap (%)", 50, 100, 85, 1)
planned_shrink_pct = st.sidebar.slider("Planned Shrinkage (%)", 0, 60, 20, 1)
unplanned_shrink_pct = st.sidebar.slider("Unplanned Shrinkage (%)", 0, 40, 8, 1)
default_aht = st.sidebar.number_input("Default AHT (sec, used if not in file)", min_value=10, value=300, step=10)

st.sidebar.subheader("3. Queueing Model")
use_erlang_a = st.sidebar.toggle("Use Erlang A (model abandonment)", value=False)

target_sl = target_sl_pct / 100.0
max_occupancy = max_occupancy_pct / 100.0
planned_shrinkage = planned_shrink_pct / 100.0
unplanned_shrinkage = unplanned_shrink_pct / 100.0

st.sidebar.divider()
st.sidebar.caption("All Erlang math runs in log-space (gammaln) — stable to 1,000+ agents.")


# --------------------------------------------------------------------------- #
# Header + Tabs
# --------------------------------------------------------------------------- #

st.title("PyErlang-ML: Forecasting & Erlang Capacity Engine")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧹 Data Ingestion & Hygiene",
    "🤖 ML Forecasting Engine",
    "📐 Erlang C/A Capacity Planning",
    "🎛️ What-If Sandbox & Export",
])


# =============================================================================
# TAB 1 — Data Ingestion & Hygiene Dashboard
# =============================================================================
with tab1:
    st.header("Data Ingestion & Hygiene Dashboard")

    if st.session_state["raw_df"] is None:
        st.info("Upload a `.csv` or `.xlsx` file in the sidebar to begin. "
                "Required columns: `timestamp`, `volume`. Optional: `aht` (seconds).")
    else:
        raw_df = st.session_state["raw_df"]
        with st.expander("Raw file preview", expanded=False):
            st.dataframe(raw_df.head(20), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            outlier_method = st.radio("Outlier detection method", ["IQR", "Z-Score"], horizontal=True)
        with col_b:
            if outlier_method == "IQR":
                iqr_mult = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
                z_thresh = 3.0
            else:
                z_thresh = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
                iqr_mult = 1.5

        if st.button("Run Data Hygiene Pipeline", type="primary"):
            try:
                cleaned, report = dh.run_hygiene_pipeline(
                    raw_df, default_aht=default_aht,
                    outlier_method="iqr" if outlier_method == "IQR" else "zscore",
                    iqr_multiplier=iqr_mult, z_threshold=z_thresh,
                )
                st.session_state["cleaned_df"] = cleaned
                st.session_state["hygiene_report"] = report
                st.session_state["interval_minutes"] = report.detected_interval_minutes
                st.success("Hygiene pipeline complete.")
            except Exception as e:
                st.error(f"Hygiene pipeline failed: {e}")

        if st.session_state["cleaned_df"] is not None:
            cleaned = st.session_state["cleaned_df"]
            report = st.session_state["hygiene_report"]

            st.subheader("Summary KPIs")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Volume", f"{report.total_volume:,.0f}")
            k2.metric("Avg Daily Volume", f"{report.avg_daily_volume:,.0f}")
            k3.metric("Peak Interval Volume", f"{report.peak_interval_volume:,.0f}")
            k4.metric("Missing Slots Fixed", f"{report.missing_slots_imputed:,}")
            k5.metric("Outliers Smoothed", f"{report.outliers_smoothed:,}")

            st.caption(f"Detected interval grain: **{report.detected_interval_minutes} minutes** "
                       f"· Rows in: {report.rows_in:,} · Rows out: {report.rows_out:,} "
                       f"· Zero-AHT intervals backfilled: {report.zero_aht_intervals:,}")
            for note in report.notes:
                st.caption(f"ℹ️ {note}")

            st.subheader("Raw vs. Cleaned Volume")
            fig = go.Figure()
            if "volume_raw" in cleaned.columns:
                fig.add_trace(go.Scatter(x=cleaned["timestamp"], y=cleaned["volume_raw"],
                                          mode="lines", name="Raw", line=dict(color="lightgray")))
            fig.add_trace(go.Scatter(x=cleaned["timestamp"], y=cleaned["volume"],
                                      mode="lines", name="Cleaned", line=dict(color="#1F4E78")))
            if "is_outlier" in cleaned.columns:
                outliers = cleaned[cleaned["is_outlier"]]
                fig.add_trace(go.Scatter(x=outliers["timestamp"], y=outliers["volume_raw"],
                                          mode="markers", name="Flagged Outlier",
                                          marker=dict(color="red", size=7, symbol="x")))
            fig.update_layout(height=420, xaxis_title="Timestamp", yaxis_title="Volume",
                               legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View cleaned data table"):
                st.dataframe(cleaned, use_container_width=True)


# =============================================================================
# TAB 2 — Advanced ML Forecasting Engine
# =============================================================================
with tab2:
    st.header("Advanced ML Forecasting Engine")

    if st.session_state["cleaned_df"] is None:
        st.info("Run the Data Hygiene pipeline in Tab 1 first.")
    else:
        cleaned = st.session_state["cleaned_df"]
        interval_minutes = st.session_state["interval_minutes"]

        horizon_choice = st.radio(
            "Forecast Horizon",
            ["Long-Term (Weekly, up to 12 months)", "Short-Term (Daily, up to 12 weeks)", "Interval-Level (30-min direct or profile split)"],
            horizontal=False,
        )

        available = mlf.available_models()
        chosen_models = st.multiselect("Models to train & compare", available, default=available)

        colH1, colH2 = st.columns(2)
        with colH1:
            if horizon_choice.startswith("Long"):
                n_periods = st.slider("Weeks ahead", 4, 52, 12)
                agg_freq, agg_label = "W", "Weekly"
            elif horizon_choice.startswith("Short"):
                n_periods = st.slider("Days ahead", 7, 84, 28)
                agg_freq, agg_label = "D", "Daily"
            else:
                n_periods_days = st.slider("Days ahead (interval-level)", 1, 28, 7)
                agg_freq, agg_label = "D", "Daily-then-split"
        with colH2:
            n_folds = st.slider("Backtest folds (expanding window)", 1, 6, 3)

        if st.button("Train Models & Backtest", type="primary"):
            with st.spinner("Aggregating series, backtesting (walk-forward), and generating forecasts..."):
                # Aggregate raw interval data up to the chosen grain for Long/Short term;
                # Interval-level uses a daily model + intraday profile split.
                ts = cleaned.set_index("timestamp")["volume"].resample(agg_freq).sum()
                ts = ts.asfreq(agg_freq).fillna(0.0)

                horizon = n_periods if not horizon_choice.startswith("Interval") else n_periods_days

                try:
                    summaries = mlf.expanding_window_backtest(ts, agg_freq, chosen_models, horizon, n_folds=n_folds)
                    forecasts = mlf.fit_final_models_and_forecast(ts, agg_freq, chosen_models, horizon, summaries)

                    st.session_state["backtest_summaries"] = summaries
                    st.session_state["forecasts"] = forecasts
                    st.session_state["forecast_horizon_label"] = horizon_choice
                    st.session_state["agg_series"] = ts
                    st.session_state["agg_freq"] = agg_freq

                    if horizon_choice.startswith("Interval"):
                        profile = mlf.build_intraday_profile(cleaned, interval_minutes)
                        st.session_state["intraday_profile"] = profile

                    st.success("Training & backtesting complete.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

        if st.session_state["backtest_summaries"]:
            st.subheader("Model Leaderboard (Expanding-Window Backtest)")
            rows = []
            for s in st.session_state["backtest_summaries"]:
                if s.fit_error:
                    rows.append({"Model": s.model_name, "WAPE": None, "MAPE": None, "RMSE": None, "MAE": None, "Note": s.fit_error})
                else:
                    rows.append({"Model": s.model_name, **s.avg_metrics, "Note": f"{len(s.folds)} fold(s)"})
            leaderboard_df = pd.DataFrame(rows).sort_values("WAPE", na_position="last")
            st.session_state["leaderboard_df"] = leaderboard_df
            st.dataframe(
                leaderboard_df.style.format({"WAPE": "{:.3f}", "MAPE": "{:.3f}", "RMSE": "{:.1f}", "MAE": "{:.1f}"}, na_rep="—"),
                use_container_width=True,
            )

            st.subheader("Actual vs. Predicted (Backtest Overlay)")
            best_model = leaderboard_df.iloc[0]["Model"] if not leaderboard_df.empty else None
            plot_model = st.selectbox("Model to inspect", [s.model_name for s in st.session_state["backtest_summaries"]],
                                       index=0 if best_model is None else [s.model_name for s in st.session_state["backtest_summaries"]].index(best_model))
            summary = next((s for s in st.session_state["backtest_summaries"] if s.model_name == plot_model), None)
            if summary and summary.folds:
                fig = go.Figure()
                ts = st.session_state["agg_series"]
                fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name="Actual (full history)", line=dict(color="lightgray")))
                for f in summary.folds:
                    idx = pd.date_range(f.test_start, periods=len(f.y_true), freq=st.session_state["agg_freq"])
                    fig.add_trace(go.Scatter(x=idx, y=f.y_pred, mode="lines+markers", name=f"Fold {f.fold+1} Predicted"))
                fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Volume")
                st.plotly_chart(fig, use_container_width=True)
            elif summary and summary.fit_error:
                st.warning(f"{plot_model}: {summary.fit_error}")

        if st.session_state["forecasts"]:
            st.subheader("Forward Forecast")
            model_names = list(st.session_state["forecasts"].keys())
            selected = st.selectbox("Forecast to use downstream (Erlang planning)", model_names, key="fc_select")
            st.session_state["chosen_forecast_model"] = selected

            fig2 = go.Figure()
            ts = st.session_state["agg_series"]
            fig2.add_trace(go.Scatter(x=ts.index[-60:], y=ts.values[-60:], mode="lines", name="History", line=dict(color="gray")))
            for name, fo in st.session_state["forecasts"].items():
                fig2.add_trace(go.Scatter(x=fo.dates, y=fo.values, mode="lines+markers", name=name,
                                           line=dict(dash="dot") if name != selected else dict(width=3)))
            fig2.update_layout(height=450, xaxis_title="Date", yaxis_title="Forecasted Volume")
            st.plotly_chart(fig2, use_container_width=True)

            if st.button("Send Selected Forecast → Erlang Capacity Planning", type="primary"):
                fo = st.session_state["forecasts"][selected]
                if st.session_state["forecast_horizon_label"].startswith("Interval"):
                    profile = st.session_state.get("intraday_profile")
                    interval_df = mlf.split_daily_forecast_to_intervals(fo, profile, interval_minutes)
                else:
                    # Long/Short-term forecasts get split to interval-level via the
                    # historical intraday profile as well, so Tab 3 always works at 30-min grain.
                    profile = mlf.build_intraday_profile(cleaned, interval_minutes)
                    # For weekly/daily forecasts, first ensure daily granularity
                    if st.session_state["agg_freq"] == "W":
                        # crude spread of weekly total evenly across 7 days, then split intraday
                        daily_dates, daily_vals = [], []
                        for d, v in zip(fo.dates, fo.values):
                            for offset in range(7):
                                daily_dates.append(d - pd.Timedelta(days=6 - offset))
                                daily_vals.append(v / 7.0)
                        daily_fo = mlf.ForecastOutput(dates=pd.DatetimeIndex(daily_dates), values=np.array(daily_vals), model_name=selected)
                    else:
                        daily_fo = fo
                    interval_df = mlf.split_daily_forecast_to_intervals(daily_fo, profile, interval_minutes)

                # attach a flat default AHT (median observed) to the interval forecast
                median_aht = cleaned["aht"].median() if "aht" in cleaned.columns else default_aht
                interval_df["aht"] = median_aht
                st.session_state["interval_forecast_df"] = interval_df
                st.success(f"{len(interval_df):,} interval rows sent to the Capacity Planning tab.")


# =============================================================================
# TAB 3 — Erlang C/A Capacity Planning Engine
# =============================================================================
with tab3:
    st.header("Erlang C/A Capacity Planning Engine")

    if st.session_state["interval_forecast_df"] is None:
        st.info("Send a forecast from Tab 2 to populate interval-level capacity planning.")
    else:
        interval_df = st.session_state["interval_forecast_df"]
        interval_seconds = st.session_state["interval_minutes"] * 60

        if st.button("Run Erlang Capacity Plan", type="primary"):
            with st.spinner("Solving Erlang staffing requirements per interval (log-space, overflow-safe)..."):
                plan_df = cp.build_staffing_plan(
                    interval_df, interval_seconds=interval_seconds, target_sl=target_sl,
                    target_answer_seconds=target_sl_time, max_occupancy=max_occupancy,
                    planned_shrinkage=planned_shrinkage, unplanned_shrinkage=unplanned_shrinkage,
                    default_aht_seconds=default_aht, use_erlang_a=use_erlang_a, patience_seconds=mta_seconds,
                )
                st.session_state["staffing_plan_df"] = plan_df
                st.success("Capacity plan generated.")

        plan_df = st.session_state["staffing_plan_df"]
        if plan_df is not None:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Peak Interval Net FTE", f"{plan_df['net_fte'].max():,.1f}")
            k2.metric("Peak Interval Gross FTE", f"{plan_df['gross_fte'].max():,.1f}")
            k3.metric("Avg Predicted SL", f"{plan_df['predicted_sl'].mean()*100:,.1f}%")
            k4.metric("Avg Predicted Occupancy", f"{plan_df['predicted_occupancy'].mean()*100:,.1f}%")

            st.subheader("Forecasted Volume vs. Required Staffing vs. Predicted SLA")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plan_df["timestamp"], y=plan_df["forecast_volume"], name="Forecast Volume", yaxis="y1", line=dict(color="#1F4E78")))
            fig.add_trace(go.Scatter(x=plan_df["timestamp"], y=plan_df["gross_fte"], name="Gross FTE Required", yaxis="y2", line=dict(color="#E67E22")))
            fig.add_trace(go.Scatter(x=plan_df["timestamp"], y=plan_df["predicted_sl"]*100, name="Predicted SL%", yaxis="y3", line=dict(color="#2ECC71", dash="dot")))
            fig.update_layout(
                height=480,
                xaxis=dict(domain=[0, 0.92]),
                yaxis=dict(title="Volume"),
                yaxis2=dict(title="Gross FTE", overlaying="y", side="right"),
                yaxis3=dict(title="SL%", overlaying="y", side="right", position=1.0, anchor="free"),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Hourly Staffing Heatmap (Day-of-Week × Hour)")
            hm = plan_df.copy()
            hm["dow"] = pd.to_datetime(hm["timestamp"]).dt.day_name()
            hm["hour"] = pd.to_datetime(hm["timestamp"]).dt.hour
            pivot = hm.pivot_table(index="dow", columns="hour", values="gross_fte", aggfunc="mean")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            pivot = pivot.reindex([d for d in dow_order if d in pivot.index])
            fig_heat = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues",
                                  labels=dict(x="Hour of Day", y="Day of Week", color="Gross FTE"))
            st.plotly_chart(fig_heat, use_container_width=True)

            with st.expander("View full interval staffing plan"):
                st.dataframe(plan_df, use_container_width=True)


# =============================================================================
# TAB 4 — What-If Sandbox & Multi-Sheet Export
# =============================================================================
with tab4:
    st.header("What-If Sandbox & Multi-Sheet Export")

    if st.session_state["interval_forecast_df"] is None or st.session_state["staffing_plan_df"] is None:
        st.info("Generate a base capacity plan in Tab 3 first.")
    else:
        base_forecast = st.session_state["interval_forecast_df"]
        base_plan = st.session_state["staffing_plan_df"]
        interval_seconds = st.session_state["interval_minutes"] * 60

        st.subheader("Scenario Controls")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            volume_delta = st.slider("Volume Surge/Drop (%)", -50, 100, 0, 1)
        with c2:
            aht_delta = st.slider("AHT Shift (%)", -30, 50, 0, 1)
        with c3:
            planned_delta = st.slider("Planned Shrinkage Spike (pp)", -10, 20, 0, 1)
        with c4:
            sl_override = st.slider("SL Target Override (%)", 50, 99, target_sl_pct, 1)

        if st.button("Run Scenario", type="primary"):
            with st.spinner("Recomputing Erlang staffing under scenario assumptions..."):
                scenario_forecast = cp.apply_whatif_scenario(
                    base_forecast, volume_pct_change=volume_delta / 100.0, aht_pct_change=aht_delta / 100.0,
                )
                scenario_planned_shrink = min(max(planned_shrinkage + planned_delta / 100.0, 0.0), 0.9)

                scenario_plan = cp.build_staffing_plan(
                    scenario_forecast, interval_seconds=interval_seconds, target_sl=sl_override / 100.0,
                    target_answer_seconds=target_sl_time, max_occupancy=max_occupancy,
                    planned_shrinkage=scenario_planned_shrink, unplanned_shrinkage=unplanned_shrinkage,
                    default_aht_seconds=default_aht, use_erlang_a=use_erlang_a, patience_seconds=mta_seconds,
                )
                st.session_state["scenario_plan_df"] = scenario_plan

        if "scenario_plan_df" in st.session_state and st.session_state["scenario_plan_df"] is not None:
            scenario_plan = st.session_state["scenario_plan_df"]
            base_gross = base_plan["gross_fte"].sum()
            scenario_gross = scenario_plan["gross_fte"].sum()
            delta_fte = scenario_gross - base_gross
            base_peak = base_plan["gross_fte"].max()
            scenario_peak = scenario_plan["gross_fte"].max()

            st.subheader("Scenario Impact")
            colx, coly, colz = st.columns(3)
            colx.metric("Total Gross FTE (Interval-Sum)", f"{scenario_gross:,.1f}", delta=f"{delta_fte:+,.1f}")
            coly.metric("Peak Interval Gross FTE", f"{scenario_peak:,.1f}", delta=f"{scenario_peak - base_peak:+,.1f}")
            colz.metric("Avg Predicted SL", f"{scenario_plan['predicted_sl'].mean()*100:.1f}%")

            direction = "surge" if volume_delta >= 0 else "drop"
            st.info(f"Under a **{abs(volume_delta)}% volume {direction}**, **{aht_delta:+d}% AHT shift**, and "
                    f"**{planned_delta:+d}pp planned shrinkage** scenario, total Gross FTE requirement changes by "
                    f"**{delta_fte:+,.1f} FTE** vs. the base plan.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=base_plan["timestamp"], y=base_plan["gross_fte"], name="Base Plan Gross FTE", line=dict(color="gray")))
            fig.add_trace(go.Scatter(x=scenario_plan["timestamp"], y=scenario_plan["gross_fte"], name="Scenario Gross FTE", line=dict(color="#E67E22")))
            fig.update_layout(height=420, xaxis_title="Timestamp", yaxis_title="Gross FTE")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("📥 Export Multi-Sheet Excel Workbook")

        export_plan = st.session_state.get("scenario_plan_df")
        if export_plan is None:
            export_plan = base_plan
            st.caption("Exporting the base capacity plan (run a scenario above to export scenario numbers instead).")
        else:
            st.caption("Exporting the most recent scenario plan.")

        n_days = max(pd.to_datetime(export_plan["timestamp"]).dt.date.nunique(), 1)
        monthly_factor = 30.0 / n_days

        kpis = {
            "Report Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Interval Grain (min)": st.session_state["interval_minutes"],
            "Target Service Level": f"{target_sl_pct}%",
            "Target Answer Time (sec)": target_sl_time,
            "Max Occupancy Cap": f"{max_occupancy_pct}%",
            "Planned Shrinkage": f"{planned_shrink_pct}%",
            "Unplanned Shrinkage": f"{unplanned_shrink_pct}%",
            "Queueing Model": "Erlang A (with abandonment)" if use_erlang_a else "Erlang C",
            "Total Forecast Volume": f"{export_plan['forecast_volume'].sum():,.0f}",
            "Peak Interval Gross FTE": f"{export_plan['gross_fte'].max():,.1f}",
            "Avg Predicted SL": f"{export_plan['predicted_sl'].mean()*100:,.1f}%",
            "Est. Monthly Staffing Budget (Gross FTE-hours, scaled)": f"{(export_plan['gross_fte'].sum() * (st.session_state['interval_minutes']/60) * monthly_factor):,.0f}",
        }

        monthly_budget_df = (
            export_plan.assign(month=pd.to_datetime(export_plan["timestamp"]).dt.to_period("M").astype(str))
            .groupby("month")
            .agg(total_volume=("forecast_volume", "sum"), avg_gross_fte=("gross_fte", "mean"), peak_gross_fte=("gross_fte", "max"))
            .reset_index()
        )

        leaderboard_df = st.session_state.get("leaderboard_df", pd.DataFrame())

        if st.button("Generate Excel Export", type="primary"):
            with st.spinner("Building multi-sheet workbook..."):
                xlsx_bytes = exp.build_excel_workbook(
                    kpis=kpis, interval_plan_df=export_plan, leaderboard_df=leaderboard_df,
                    monthly_budget_df=monthly_budget_df,
                )
                st.session_state["export_bytes"] = xlsx_bytes

        if "export_bytes" in st.session_state:
            st.download_button(
                "⬇️ Download PyErlang-ML_Capacity_Plan.xlsx",
                data=st.session_state["export_bytes"],
                file_name=f"PyErlang-ML_Capacity_Plan_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
