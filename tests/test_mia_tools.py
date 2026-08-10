import json

import pytest

from energy_forecast.mia.registry import Session
from energy_forecast.mia.schema import TOOL_SCHEMAS
from energy_forecast.mia.tools import (
    MAX_PREVIEW_ROWS,
    TOOL_HANDLERS,
    _handle_calculate_error_slices,
    _handle_run_backtest,
    _handle_run_dispatch_scenario,
)


@pytest.fixture
def session(synthetic_raw_demand):
    return Session(raw_data=synthetic_raw_demand(14))


def test_run_backtest_returns_json_safe_summary(session):
    args = {"forecast_fn_id": "seasonal_naive", "split_date": "2024-01-11"}

    response = _handle_run_backtest(session, args)

    json.dumps(response)  # must not raise
    assert "predictions_id" in response
    assert "preview" not in response


def test_run_backtest_unknown_forecast_fn_id_raises(session):
    args = {"forecast_fn_id": "nope", "split_date": "2024-01-11"}

    with pytest.raises(KeyError):
        _handle_run_backtest(session, args)


def test_run_backtest_preview_requires_a_window(session):
    args = {"forecast_fn_id": "seasonal_naive", "split_date": "2024-01-11", "return_predictions": True}

    response = _handle_run_backtest(session, args)

    assert "preview" not in response


def test_run_backtest_preview_with_window_is_included_and_bounded(session):
    args = {
        "forecast_fn_id": "seasonal_naive",
        "split_date": "2024-01-11",
        "return_predictions": True,
        "preview_start": "2024-01-11",
        "preview_end": "2024-01-11",
    }

    response = _handle_run_backtest(session, args)

    assert "preview" in response
    assert 0 < len(response["preview"]) <= MAX_PREVIEW_ROWS
    json.dumps(response)


def test_calculate_error_slices_unknown_predictions_id_raises(session):
    with pytest.raises(KeyError):
        _handle_calculate_error_slices(session, {"predictions_id": "pred_999"})


def test_calculate_error_slices_end_to_end(session):
    backtest_response = _handle_run_backtest(
        session, {"forecast_fn_id": "seasonal_naive", "split_date": "2024-01-11"}
    )

    slices = _handle_calculate_error_slices(session, {"predictions_id": backtest_response["predictions_id"]})

    assert set(slices.keys()) == {"by_time_of_day", "by_day_of_week", "by_month"}
    json.dumps(slices)


def test_run_dispatch_scenario_consumes_solve_budget(session):
    backtest_response = _handle_run_backtest(
        session, {"forecast_fn_id": "seasonal_naive", "split_date": "2024-01-11"}
    )

    dispatch_response = _handle_run_dispatch_scenario(
        session, {"predictions_id": backtest_response["predictions_id"]}
    )

    json.dumps(dispatch_response)
    assert "total_cost" in dispatch_response
    assert "schedule_id" in dispatch_response
    # test window is 2024-01-11 through 2024-01-14 inclusive: 4 calendar days
    assert session.dispatch_solves_used == 4


def test_run_dispatch_scenario_rejects_when_budget_exceeded(session):
    backtest_response = _handle_run_backtest(
        session, {"forecast_fn_id": "seasonal_naive", "split_date": "2024-01-11"}
    )
    session.dispatch_solves_budget = 1

    with pytest.raises(ValueError):
        _handle_run_dispatch_scenario(session, {"predictions_id": backtest_response["predictions_id"]})

    assert session.dispatch_solves_used == 0


def test_run_dispatch_scenario_unknown_predictions_id_raises(session):
    with pytest.raises(KeyError):
        _handle_run_dispatch_scenario(session, {"predictions_id": "pred_999"})


def test_tool_handler_names_match_schema_names():
    schema_names = {tool["name"] for tool in TOOL_SCHEMAS}
    handler_names = set(TOOL_HANDLERS.keys())

    assert schema_names == handler_names
