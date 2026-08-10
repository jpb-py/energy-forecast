import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_raw_demand():
    """Factory fixture producing deterministic synthetic demand data shaped like
    energy_forecast.data.load_demand_data's output (tz-aware UTC datetime index,
    SETTLEMENT_PERIOD and ND columns), with daily + weekly seasonality so lag
    features are meaningful, built directly rather than round-tripping a CSV."""

    def _make(n_days: int = 14) -> pd.DataFrame:
        periods = n_days * 48
        index = pd.date_range("2024-01-01", periods=periods, freq="30min", tz="UTC")
        settlement_period = (np.arange(periods) % 48) + 1
        demand = 100 + 10 * np.sin(2 * np.pi * settlement_period / 48) + 0.01 * np.arange(periods)

        return pd.DataFrame(
            {"SETTLEMENT_PERIOD": settlement_period, "ND": demand},
            index=index,
        )

    return _make
