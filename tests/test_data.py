from pathlib import Path
import pandas as pd
from energy_forecast.data import load_demand_data


def test_load_demand_data_ordinary_day_shape(tmp_path):
    """Sanity check: 48 periods should produce
    48 unique, monotonically increasing half-hourly timestamps starting
    at midnight.
    """
    raw = pd.DataFrame({
        "SETTLEMENT_DATE": ["15-Jun-2024"] * 48,
        "SETTLEMENT_PERIOD": range(1, 49),
        "ND": range(48),
    })
    csv_path = tmp_path / "demand.csv"
    raw.to_csv(csv_path, index=False)

    df = load_demand_data(csv_path)

    assert len(df) == 48
    assert df.index.is_unique
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("2024-06-15 00:00")
    assert df.index[-1] == pd.Timestamp("2024-06-15 23:30")