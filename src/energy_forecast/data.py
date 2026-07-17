from pathlib import Path
import pandas as pd


def load_demand_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'], format='%d-%b-%Y')

    # Convert to UTC before creating single datetime column to avoid BST issues
    utc_anchor = df['SETTLEMENT_DATE'].dt.tz_localize('Europe/London').dt.tz_convert('UTC')
    df['datetime'] = utc_anchor + pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')

    df = df.set_index('datetime').sort_index()

    return df

