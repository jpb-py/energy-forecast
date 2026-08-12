import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import HierarchicalSigmaPriors
from energy_forecast.features import build_features
from energy_forecast.forecast import (
    bayesian_hierarchical_forecast,
    forecast_with_interval,
    naive_seasonal_forecast,
)

SPLIT_DATE = pd.Timestamp("2024-01-11", tz="UTC")

# Small MCMC settings throughout this file's Bayesian tests to keep runtime low; these are
# smoke/contract tests, not checks of posterior accuracy.
FAST_PRIORS = HierarchicalSigmaPriors(draws=100, tune=100, chains=2)


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


def test_bayesian_hierarchical_forecast_output_shape(synthetic_raw_demand):
    df = build_features(synthetic_raw_demand(14))

    forecast = bayesian_hierarchical_forecast(df, SPLIT_DATE, priors=FAST_PRIORS, random_seed=0)

    test_df = df[df.index >= SPLIT_DATE]
    assert list(forecast.columns) == ["Forecast", "Lower Forecast", "Upper Forecast"]
    assert forecast.index.equals(test_df.index)
    assert len(forecast) == len(test_df)
    assert forecast.notna().all().all()
    assert (forecast["Lower Forecast"] <= forecast["Forecast"]).all()
    assert (forecast["Forecast"] <= forecast["Upper Forecast"]).all()


def test_bayesian_hierarchical_forecast_same_point_forecast_as_forecast_with_interval(synthetic_raw_demand):
    # The Bayesian model is a fixed plug-in on the mean (see module docstring in forecast.py) --
    # only the interval should differ, never the point forecast.
    df = build_features(synthetic_raw_demand(14))

    linear = forecast_with_interval(df, SPLIT_DATE)
    bayesian = bayesian_hierarchical_forecast(df, SPLIT_DATE, priors=FAST_PRIORS, random_seed=0)

    np.testing.assert_allclose(bayesian["Forecast"].to_numpy(), linear["Forecast"].to_numpy())


def test_bayesian_hierarchical_forecast_interval_width_varies_by_hour(synthetic_raw_demand):
    # The shared synthetic_raw_demand fixture is purely deterministic (no injected noise), so an
    # OLS fit using y_lag_336 recovers it almost exactly and leaves no real heteroscedasticity to
    # detect. Inject hour-dependent noise on top of it here so there's an actual signal for the
    # hierarchical sigma model to pick up.
    raw = synthetic_raw_demand(28)
    local_hour = raw.index.tz_convert("Europe/London").hour
    noise_scale = np.where(local_hour < 12, 0.5, 5.0)
    rng = np.random.default_rng(0)
    raw = raw.copy()
    raw["ND"] = raw["ND"] + rng.normal(0.0, noise_scale)

    df = build_features(raw)
    split_date = pd.Timestamp("2024-01-20", tz="UTC")
    priors = HierarchicalSigmaPriors(draws=300, tune=300, chains=2)

    forecast = bayesian_hierarchical_forecast(df, split_date, priors=priors, random_seed=0)

    test_df = df[df.index >= split_date]
    test_local_hour = test_df.index.tz_convert("Europe/London").hour
    width = forecast["Upper Forecast"] - forecast["Lower Forecast"]

    morning_width = width[test_local_hour < 12].mean()
    evening_width = width[test_local_hour >= 12].mean()

    assert evening_width > morning_width * 1.5
