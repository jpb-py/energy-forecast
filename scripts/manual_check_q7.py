"""Manual, one-off check of the corrected run_dispatch_scenario logic (see the
"seasonal_naive looks cheaper than linear_lagged" investigation).

Reruns the Nov 1-7 and Nov 8-14 2024 dispatch comparison for both registered
forecast models, calling the underlying functions directly -- no agent loop,
no LLM. total_profit is directly comparable to the old (pre-fix) "total_cost"
numbers, since that's the same forecast-vs-itself figure, just renamed.
realized_total_profit is new: it re-prices the same schedule against actual
demand instead, which is what should be trusted for comparing models.
"""

from pathlib import Path

import pandas as pd

from energy_forecast.data import load_demand_data
from energy_forecast.evaluation import run_backtest
from energy_forecast.mia.registry import FORECAST_FUNCTIONS, Session
from energy_forecast.mia.tools import _handle_run_dispatch_scenario

DATA_PATH = Path("data/demanddata_2024.csv")

MODEL_IDS = ["linear_lagged", "seasonal_naive"]

WEEKS = [
    ("Nov 1-7", pd.Timestamp("2024-11-01", tz="UTC"), pd.Timestamp("2024-11-07 23:30", tz="UTC")),
    ("Nov 8-14", pd.Timestamp("2024-11-08", tz="UTC"), pd.Timestamp("2024-11-14 23:30", tz="UTC")),
]


def main() -> None:
    raw_data = load_demand_data(DATA_PATH)
    results = []

    for model_id in MODEL_IDS:
        forecast_fn = FORECAST_FUNCTIONS[model_id]

        for week_label, split_date, end_date in WEEKS:
            # 1. Backtest the model over the week to get its predictions
            backtest = run_backtest(
                forecast_fn, split_date, raw_data, return_predictions=True, end_date=end_date
            )
            predictions = backtest["predictions"]

            # 2. Independently pull actual demand for the same week straight from
            #    raw_data, and cross-check it against predictions["Actual"]
            #    (built the same way inside run_backtest, just via features_df).
            actual_demand = raw_data.loc[split_date:end_date, "ND"]
            pd.testing.assert_series_equal(actual_demand, predictions["Actual"], check_names=False)

            # 3. Run the corrected run_dispatch_scenario tool logic directly
            #    (same code path the agent calls, just without the agent loop).
            session = Session(raw_data=raw_data)
            predictions_id = session.store.put("pred", predictions)
            dispatch = _handle_run_dispatch_scenario(session, {"predictions_id": predictions_id})

            result = {
                "model": model_id,
                "week": week_label,
                "mae": backtest["mae"],
                "rmse": backtest["rmse"],
                "total_profit": dispatch["total_profit"],
                "realized_total_profit": dispatch["realized_total_profit"],
            }
            results.append(result)

            print(
                f"{model_id:15s} {week_label:10s} "
                f"mae={result['mae']:8.1f} rmse={result['rmse']:8.1f} "
                f"total_profit={result['total_profit']:9.2f} "
                f"realized_total_profit={result['realized_total_profit']:9.2f}"
            )

    print()
    header = f"{'Model':15s} {'Week':10s} {'MAE':>8s} {'RMSE':>8s} {'Total Profit':>13s} {'Realized Profit':>16s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:15s} {r['week']:10s} {r['mae']:8.1f} {r['rmse']:8.1f} "
            f"{r['total_profit']:13.2f} {r['realized_total_profit']:16.2f}"
        )


if __name__ == "__main__":
    main()
