import numpy as np
import pandas as pd
import pytest

from energy_forecast.features import build_features
from energy_forecast.forecast import forecast_with_interval, naive_seasonal_forecast

SPLIT_DATE = pd.Timestamp("2024-01-11", tz="UTC")


def test_naive_seasonal_forecast_matches_lag_336(synthetic_raw_demand):
    df = build_features(synthetic_raw_demand(14))

    forecast = naive_seasonal_forecast(df, SPLIT_DATE)

    test_df = df[df.index >= SPLIT_DATE]
    assert list(forecast.columns) == ["Forecast", "Lower Forecast", "Upper Forecast"]
    assert forecast.index.equals(test_df.index)
    np.testing.assert_allclose(forecast["Forecast"].to_numpy(), test_df["y_lag_336"].to_numpy())

    # Interval should bracket the point forecast symmetrically
    np.testing.assert_allclose(
        (forecast["Upper Forecast"] - forecast["Forecast"]).to_numpy(),
        (forecast["Forecast"] - forecast["Lower Forecast"]).to_numpy(),
    )


def test_forecast_with_interval_output_shape(synthetic_raw_demand):
    df = build_features(synthetic_raw_demand(14))

    forecast = forecast_with_interval(df, SPLIT_DATE)

    test_df = df[df.index >= SPLIT_DATE]
    assert list(forecast.columns) == ["Forecast", "Lower Forecast", "Upper Forecast"]
    assert forecast.index.equals(test_df.index)
    assert len(forecast) == len(test_df)
    assert forecast.notna().all().all()


def test_forecast_with_interval_raises_on_insufficient_training_rows(synthetic_raw_demand):
    df = build_features(synthetic_raw_demand(14))
    split_date = df.index.min()  # no rows exist before the first valid (post-lag-warmup) row

    with pytest.raises(ValueError, match="Not enough training rows"):
        forecast_with_interval(df, split_date)
