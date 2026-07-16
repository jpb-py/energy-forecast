from pathlib import Path
import pandas as pd

def load_demand_data(path: Path = Path("data/demanddata_2024.csv")) -> pd.DataFrame:
    """Load NESO half-hourly demand data and index by datetime"""
    df = pd.read_csv(path)

    # Parse the date and create a proper datetime index
    df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'], format='%d-%b-%Y')

    # Create a single datetime column combining date and settlement period
    # Each period is 30 minutes, period 1 starts at 00:00
    df['datetime'] = df['SETTLEMENT_DATE'] + pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')

    df = df.set_index('datetime')
    df = df.sort_index()
    return df

