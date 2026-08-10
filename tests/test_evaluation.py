import pandas as pd
import pytest

from energy_forecast.evaluation import (
    calculate_error_slices,
    compare_model_versions,
    run_backtest,
)

SPLIT_DATE = pd.Timestamp("2024-01-11", tz="UTC")


def _perfect_forecast(df: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    test_df = df[df.index >= split_date]
    forecast = test_df["ND"]
    return pd.DataFrame(
        {"Forecast": forecast, "Lower Forecast": forecast - 1, "Upper Forecast": forecast + 1}
    )


def _offset_forecast(df: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    test_df = df[df.index >= split_date]
    forecast = test_df["ND"] + 5.0
    return pd.DataFrame(
        {"Forecast": forecast, "Lower Forecast": forecast - 1, "Upper Forecast": forecast + 1}
    )


def test_run_backtest_zero_error_for_perfect_forecast(synthetic_raw_demand):
    raw = synthetic_raw_demand(14)

    result = run_backtest(_perfect_forecast, SPLIT_DATE, raw)

    assert result["mae"] == pytest.approx(0.0, abs=1e-9)
    assert result["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert "predictions" not in result


def test_run_backtest_constant_offset_error(synthetic_raw_demand):
    raw = synthetic_raw_demand(14)

    result = run_backtest(_offset_forecast, SPLIT_DATE, raw, return_predictions=True)

    assert result["mae"] == pytest.approx(5.0, abs=1e-9)
    assert result["rmse"] == pytest.approx(5.0, abs=1e-9)
    assert "Actual" in result["predictions"].columns


def test_compare_model_versions_deltas(synthetic_raw_demand):
    raw = synthetic_raw_demand(14)

    result = compare_model_versions(_perfect_forecast, _offset_forecast, SPLIT_DATE, raw)

    assert result["a"]["mae"] == pytest.approx(0.0, abs=1e-9)
    assert result["b"]["mae"] == pytest.approx(5.0, abs=1e-9)
    assert result["mae_change"] == pytest.approx(5.0, abs=1e-9)
    assert result["rmse_change"] == pytest.approx(5.0, abs=1e-9)


def test_run_backtest_end_date_bounds_the_window(synthetic_raw_demand):
    raw = synthetic_raw_demand(14)
    end_date = pd.Timestamp("2024-01-12 23:30", tz="UTC")  # last period of the 2nd test day

    result = run_backtest(_perfect_forecast, SPLIT_DATE, raw, return_predictions=True, end_date=end_date)

    # without end_date the window runs 2024-01-11 through 2024-01-14 (4 days = 192 rows)
    assert len(result["predictions"]) == 96
    assert result["predictions"].index.min() >= SPLIT_DATE
    assert result["predictions"].index.max() <= end_date


def test_compare_model_versions_end_date_threads_through_both_sides(synthetic_raw_demand):
    raw = synthetic_raw_demand(14)
    end_date = pd.Timestamp("2024-01-12 23:30", tz="UTC")

    result = compare_model_versions(
        _perfect_forecast, _offset_forecast, SPLIT_DATE, raw, return_predictions=True, end_date=end_date
    )

    assert len(result["a"]["predictions"]) == 96
    assert len(result["b"]["predictions"]) == 96


def test_calculate_error_slices_bucket_stats():
    index = pd.date_range("2024-01-01", periods=4, freq="12h", tz="UTC")
    predictions = pd.DataFrame(
        {"Forecast": [10.0, 10.0, 10.0, 10.0], "Actual": [8.0, 12.0, 8.0, 12.0]}, index=index
    )

    slices = calculate_error_slices(predictions)

    assert set(slices.keys()) == {"by_time_of_day", "by_day_of_week", "by_month"}
    # errors (Forecast - Actual) are [2, -2, 2, -2]; every group here contains one
    # +2 and one -2 sample, so bias should cancel to 0 while mae stays at 2
    for group_stats in slices["by_day_of_week"].values():
        assert group_stats["mae"] == pytest.approx(2.0)
        assert group_stats["bias"] == pytest.approx(0.0)

    for group_stats in slices["by_time_of_day"].values():
        assert group_stats["rmse"] == pytest.approx(2.0)
