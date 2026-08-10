import numpy as np
import pandas as pd

from energy_forecast.features import build_features


def run_backtest(
    forecast_fn,
    split_date: pd.Timestamp,
    raw_data: pd.DataFrame,
    return_predictions: bool = False,
) -> dict:
    features_df = build_features(raw_data)

    predictions = forecast_fn(features_df, split_date).copy()
    predictions['Actual'] = features_df.loc[predictions.index, 'ND']

    error = predictions['Forecast'] - predictions['Actual']
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    result = {'split_date': split_date, 'mae': mae, 'rmse': rmse}
    if return_predictions:
        result['predictions'] = predictions

    return result


def compare_model_versions(
    forecast_fn_a,
    forecast_fn_b,
    split_date: pd.Timestamp,
    raw_data: pd.DataFrame,
    return_predictions: bool = False,
) -> dict:
    result_a = run_backtest(forecast_fn_a, split_date, raw_data, return_predictions)
    result_b = run_backtest(forecast_fn_b, split_date, raw_data, return_predictions)

    return {
        'split_date': split_date,
        'a': result_a,
        'b': result_b,
        'rmse_change': result_b['rmse'] - result_a['rmse'],
        'mae_change': result_b['mae'] - result_a['mae'],
    }


def calculate_error_slices(predictions: pd.DataFrame) -> dict:
    # predictions must carry an 'Actual' column alongside 'Forecast', as produced by run_backtest
    local_index = predictions.index.tz_convert('Europe/London')
    error = predictions['Forecast'] - predictions['Actual']

    def _slice(labels) -> dict:
        frame = pd.DataFrame({'error': error.to_numpy(), 'label': labels})
        buckets = {}
        for label, group in frame.groupby('label', observed=True):
            e = group['error']
            buckets[label] = {
                'mae': float(np.mean(np.abs(e))),
                'rmse': float(np.sqrt(np.mean(e**2))),
                'bias': float(np.mean(e)),
            }
        return buckets

    return {
        'by_time_of_day': _slice(local_index.strftime('%H:%M')),
        'by_day_of_week': _slice(local_index.day_name()),
        'by_month': _slice(local_index.month_name()),
    }
