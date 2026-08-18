# PyErlang-ML

Enterprise Call Volume Forecasting & Erlang (C/A) Capacity Planning Engine, built with Streamlit.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## File layout

- `app.py` — Streamlit main layout, sidebar global settings, and 4-tab orchestrator.
- `modules/data_hygiene.py` — schema parsing, interval-grain auto-detect, seasonal-median missing-slot imputation, IQR/Z-score outlier smoothing.
- `modules/ml_forecaster.py` — Prophet / XGBoost / SARIMAX / Holt-Winters / weighted-Ensemble model zoo with expanding-window (walk-forward) backtesting and WAPE/MAPE/RMSE/MAE reporting.
- `modules/erlang_engine.py` — log-space (`scipy.special.gammaln`) Erlang C and Erlang A calculators, stable to 1,000+ agents.
- `modules/capacity_planner.py` — occupancy-cap solver and compound-shrinkage Net→Gross FTE conversion.
- `modules/exporter.py` — multi-sheet Excel workbook builder (Summary_Dashboard, 30Min_Interval_Plan, Model_Leaderboard).

## Input file schema

| column      | required | notes                                   |
|-------------|----------|------------------------------------------|
| `timestamp` | yes      | any parseable datetime                   |
| `volume`    | yes      | contacts offered in the interval         |
| `aht`       | no       | seconds; defaults to sidebar value if absent |

## Notes on numerical stability

All Erlang C/A math is computed in log-space using `scipy.special.gammaln` and
`np.logaddexp` / `np.logaddexp.reduce`, so it never overflows even at very
high agent counts (tested to 900+ agents / 800+ Erlangs of traffic).

## Notes on time-series validity

Backtesting exclusively uses **expanding-window (walk-forward)** splits —
never random K-Fold — to avoid look-ahead bias.
