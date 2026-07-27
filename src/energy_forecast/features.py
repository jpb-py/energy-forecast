# 47 Time dummies and therefore constant must be added later
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    # Add indexes for hour, day of week, month, and boolean for weekend day
    # Convert index to local time for calculations and then convert back
    df = df.copy() # Create copy to avoid acting on original df
    local_index = df.index.tz_convert("Europe/London")
    df['hour'] = local_index.hour
    df['day_of_week'] = local_index.day_of_week
    df['month'] = local_index.month
    df['is_weekend'] = df['day_of_week'] >= 5
    

    
    # Add lags
    y = df['ND']
    df['y_lag_1'] = y.shift(1)
    df['y_lag_48'] = y.shift(48)
    df['y_lag_336'] = y.shift(336)
    df = df.dropna(subset=['y_lag_1', 'y_lag_48', 'y_lag_336'])

    # Add time dummies (True/False entries for regression) to full df 
    # Drop first entry so don't add constant silently
    time_dummies = pd.get_dummies(df['SETTLEMENT_PERIOD'], prefix = 'period', drop_first = True)
    time_dummies.index = df.index

    df_full = pd.concat([df, time_dummies], axis = 1)
    return df_full
    


