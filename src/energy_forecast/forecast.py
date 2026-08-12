# Create forecast with time period dummies and lags.  Include prediction interval at 95%
# Split date is first date in test data
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from energy_forecast.bayesian_interval import (
    build_hierarchical_sigma_model,
    fit_hierarchical_sigma_model,
    predict_interval,
)
from energy_forecast.config import HierarchicalSigmaPriors


def _fit_mean_model(df: pd.DataFrame, split_date: pd.Timestamp):
    # Split data into training and test data
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]

    # Extract X that will use for regression
    feature_cols = ['y_lag_1','y_lag_48','y_lag_336'] + [col for col in df.columns if col.startswith('period')]

    # LinearRegression needs strictly more training rows than columns to fit at all; with the
    # lag features' own warmup already consuming the first 336 rows of df, an early split_date
    # can leave too few (or zero) training rows. Catch that here rather than deep inside sklearn.
    if len(train_df) <= len(feature_cols):
        raise ValueError(
            f"Not enough training rows before split_date={split_date} to fit linear_lagged: "
            f"{len(train_df)} available but {len(feature_cols)} feature columns (lags + "
            "settlement-period dummies) require more rows than that to fit. Choose a later "
            "split_date or provide more history."
        )

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Do regression with sklearn
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, train_df['ND'])

    y_pred_train = pd.Series(model.predict(X_train), index = X_train.index)
    y_fore = pd.Series(model.predict(X_test), index = X_test.index)

    return train_df, test_df, y_pred_train, y_fore


def forecast_with_interval(df: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    train_df, _test_df, y_pred_train, y_fore = _fit_mean_model(df, split_date)

    y_resid = train_df['ND'] - y_pred_train
    std_dev = np.std(y_resid)

    y_lower = y_fore - 1.96*std_dev
    y_upper = y_fore + 1.96*std_dev

    forecast = pd.concat([y_fore, y_lower, y_upper], axis =1, keys = ['Forecast', 'Lower Forecast', 'Upper Forecast'])

    return forecast


def bayesian_hierarchical_forecast(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    priors: HierarchicalSigmaPriors | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    # Same point forecast as forecast_with_interval; only the interval differs. The mean model
    # is treated as a fixed plug-in here — residual diagnostics found no hour-of-day bias in the
    # mean, only in the spread, so only the spread is modelled hierarchically (see
    # notebooks/04_bayesian_intervals.ipynb).
    priors = priors if priors is not None else HierarchicalSigmaPriors()
    train_df, test_df, y_pred_train, y_fore = _fit_mean_model(df, split_date)

    y_resid = train_df['ND'] - y_pred_train

    model = build_hierarchical_sigma_model(y_resid, train_df['hour'], priors)
    trace = fit_hierarchical_sigma_model(model, priors, random_seed=random_seed)
    offsets = predict_interval(model, trace, test_df['hour'], priors, random_seed=random_seed)

    y_lower = y_fore + offsets['lower_offset']
    y_upper = y_fore + offsets['upper_offset']

    forecast = pd.concat([y_fore, y_lower, y_upper], axis=1, keys=['Forecast', 'Lower Forecast', 'Upper Forecast'])

    return forecast


def naive_seasonal_forecast(df: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    # Seasonal-naive baseline: forecast = demand at the same settlement period one week ago
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]

    y_resid = train_df['ND'] - train_df['y_lag_336']
    std_dev = np.std(y_resid)

    y_fore = test_df['y_lag_336']
    y_lower = y_fore - 1.96*std_dev
    y_upper = y_fore + 1.96*std_dev

    forecast = pd.concat([y_fore, y_lower, y_upper], axis =1, keys = ['Forecast', 'Lower Forecast', 'Upper Forecast'])

    return forecast
