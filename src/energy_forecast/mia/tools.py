import pandas as pd

from energy_forecast.config import BatteryParams
from energy_forecast.dispatch import realized_profit as _realized_profit
from energy_forecast.dispatch import run_multi_day_dispatch
from energy_forecast.evaluation import (
    calculate_error_slices as _calculate_error_slices,
)
from energy_forecast.evaluation import (
    compare_model_versions as _compare_model_versions,
)
from energy_forecast.evaluation import (
    run_backtest as _run_backtest,
)
from energy_forecast.mia.registry import Session, resolve_forecast_fn

MAX_PREVIEW_ROWS = 336
DISPATCH_PARAMS = BatteryParams()


def _preview_rows(df: pd.DataFrame, start, end) -> list:
    window = df.loc[start:end]

    if len(window) > MAX_PREVIEW_ROWS:
        raise ValueError(
            f"Requested preview window has {len(window)} rows; max is {MAX_PREVIEW_ROWS}. "
            "Narrow preview_start/preview_end."
        )

    return [
        {"timestamp": timestamp.isoformat(), **{col: float(val) for col, val in row.items()}}
        for timestamp, row in window.iterrows()
    ]


def _wants_preview(args: dict, flag_name: str) -> bool:
    has_window = bool(args.get("preview_start")) or bool(args.get("preview_end"))
    return bool(args.get(flag_name, False)) and has_window


def _parse_date(value: str) -> pd.Timestamp:
    # raw_data (and everything derived from it) is UTC-indexed; a bare ISO date
    # from tool JSON is tz-naive, so it must be localized before it can be
    # compared against that index.
    date = pd.Timestamp(value)
    if date.tzinfo is None:
        date = date.tz_localize("UTC")
    return date


def _parse_end_date(value: str) -> pd.Timestamp:
    # A bare end_date (e.g. "2024-01-12") parses to that day's midnight, which
    # would exclude almost the entire day from a "<=" comparison. Extend it to
    # the last instant of that day so end_date reads as "through end_date",
    # mirroring how split_date already reads as "from the start of split_date".
    end_date = _parse_date(value)
    return end_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def _handle_run_backtest(session: Session, args: dict) -> dict:
    forecast_fn = resolve_forecast_fn(args["forecast_fn_id"])
    split_date = _parse_date(args["split_date"])
    end_date = _parse_end_date(args["end_date"]) if args.get("end_date") else None

    result = _run_backtest(
        forecast_fn, split_date, session.raw_data, return_predictions=True, end_date=end_date
    )
    predictions_id = session.store.put("pred", result["predictions"])

    response = {
        "split_date": str(split_date),
        "mae": result["mae"],
        "rmse": result["rmse"],
        "predictions_id": predictions_id,
    }
    if end_date is not None:
        response["end_date"] = args["end_date"]

    if _wants_preview(args, "return_predictions"):
        response["preview"] = _preview_rows(
            result["predictions"], args.get("preview_start"), args.get("preview_end")
        )

    return response


def _handle_compare_model_versions(session: Session, args: dict) -> dict:
    forecast_fn_a = resolve_forecast_fn(args["forecast_fn_id_a"])
    forecast_fn_b = resolve_forecast_fn(args["forecast_fn_id_b"])
    split_date = _parse_date(args["split_date"])
    end_date = _parse_end_date(args["end_date"]) if args.get("end_date") else None
    include_preview = _wants_preview(args, "return_predictions")

    result = _compare_model_versions(
        forecast_fn_a,
        forecast_fn_b,
        split_date,
        session.raw_data,
        return_predictions=True,
        end_date=end_date,
    )

    def _side(key: str) -> dict:
        side_result = result[key]
        predictions_id = session.store.put("pred", side_result["predictions"])
        side = {
            "mae": side_result["mae"],
            "rmse": side_result["rmse"],
            "predictions_id": predictions_id,
        }
        if include_preview:
            side["preview"] = _preview_rows(
                side_result["predictions"], args.get("preview_start"), args.get("preview_end")
            )
        return side

    response = {
        "split_date": str(split_date),
        "a": _side("a"),
        "b": _side("b"),
        "rmse_change": result["rmse_change"],
        "mae_change": result["mae_change"],
    }
    if end_date is not None:
        response["end_date"] = args["end_date"]

    return response


def _handle_calculate_error_slices(session: Session, args: dict) -> dict:
    predictions = session.store.get(args["predictions_id"])
    return _calculate_error_slices(predictions)


def _handle_run_dispatch_scenario(session: Session, args: dict) -> dict:
    predictions_id = args["predictions_id"]
    predictions = session.store.get(predictions_id)
    demand = predictions["Forecast"]

    if len(demand) % 48 != 0:
        raise ValueError(
            f"predictions_id '{predictions_id}' covers {len(demand)} periods, not a whole "
            "number of days (a multiple of 48)."
        )

    n_days = len(demand) // 48
    remaining = session.dispatch_solves_budget - session.dispatch_solves_used
    if n_days > remaining:
        raise ValueError(
            f"This scenario needs {n_days} daily dispatch solves but only {remaining} remain "
            f"in this investigation's budget of {session.dispatch_solves_budget}. Narrow the "
            "date range."
        )

    schedule = run_multi_day_dispatch(demand, DISPATCH_PARAMS)
    session.dispatch_solves_used += n_days

    total_profit = float(schedule["profit per period"].sum())

    schedule["realized profit per period"] = _realized_profit(schedule, predictions["Actual"], DISPATCH_PARAMS)
    realized_total_profit = float(schedule["realized profit per period"].sum())

    schedule_id = session.store.put("sched", schedule)

    response = {
        "total_profit": total_profit,
        "realized_total_profit": realized_total_profit,
        "schedule_id": schedule_id,
    }

    if _wants_preview(args, "return_schedule"):
        response["preview"] = _preview_rows(
            schedule, args.get("preview_start"), args.get("preview_end")
        )

    return response


TOOL_HANDLERS = {
    "run_backtest": _handle_run_backtest,
    "compare_model_versions": _handle_compare_model_versions,
    "calculate_error_slices": _handle_calculate_error_slices,
    "run_dispatch_scenario": _handle_run_dispatch_scenario,
}
