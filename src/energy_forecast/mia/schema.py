from energy_forecast.mia.registry import FORECAST_FUNCTIONS

_FORECAST_FN_IDS = ", ".join(sorted(FORECAST_FUNCTIONS))

TOOL_SCHEMAS = [
    {
        "name": "run_backtest",
        "description": (
            "Backtest a registered forecast function against actual demand on and after "
            "split_date, optionally bounded by end_date. Returns MAE/RMSE plus a "
            "predictions_id you can pass to calculate_error_slices or run_dispatch_scenario."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "forecast_fn_id": {
                    "type": "string",
                    "description": f"Registered forecast function id. One of: {_FORECAST_FN_IDS}.",
                },
                "split_date": {
                    "type": "string",
                    "description": "ISO date (e.g. '2024-11-01'). The test period starts here.",
                },
                "end_date": {
                    "type": "string",
                    "description": (
                        "Optional ISO date; the test period runs through the end of this day "
                        "(inclusive). If omitted, the backtest runs to the end of the dataset."
                    ),
                },
                "return_predictions": {
                    "type": "boolean",
                    "description": (
                        "If true and paired with preview_start/preview_end, include a small "
                        "inline preview of actual-vs-forecast rows for that window alongside "
                        "predictions_id. If true with no window, only predictions_id is returned."
                    ),
                },
                "preview_start": {
                    "type": "string",
                    "description": "ISO date bounding the preview window (only used if return_predictions is true).",
                },
                "preview_end": {
                    "type": "string",
                    "description": "ISO date bounding the preview window (only used if return_predictions is true).",
                },
            },
            "required": ["forecast_fn_id", "split_date"],
        },
    },
    {
        "name": "compare_model_versions",
        "description": (
            "Backtest two registered forecast functions over the same split_date and compare "
            "their MAE/RMSE. Internally runs run_backtest twice, once per model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "forecast_fn_id_a": {
                    "type": "string",
                    "description": f"Registered forecast function id. One of: {_FORECAST_FN_IDS}.",
                },
                "forecast_fn_id_b": {
                    "type": "string",
                    "description": f"Registered forecast function id. One of: {_FORECAST_FN_IDS}.",
                },
                "split_date": {
                    "type": "string",
                    "description": "ISO date. The test period starts here for both models.",
                },
                "end_date": {
                    "type": "string",
                    "description": (
                        "Optional ISO date; the test period runs through the end of this day "
                        "(inclusive), applied to both models. If omitted, the backtest runs "
                        "to the end of the dataset."
                    ),
                },
                "return_predictions": {
                    "type": "boolean",
                    "description": "As in run_backtest, applies to both a and b.",
                },
                "preview_start": {
                    "type": "string",
                    "description": "ISO date bounding the preview window (only used if return_predictions is true).",
                },
                "preview_end": {
                    "type": "string",
                    "description": "ISO date bounding the preview window (only used if return_predictions is true).",
                },
            },
            "required": ["forecast_fn_id_a", "forecast_fn_id_b", "split_date"],
        },
    },
    {
        "name": "calculate_error_slices",
        "description": (
            "Break down forecast error (mae, rmse, bias = mean(forecast - actual); positive "
            "bias means over-forecasting) by time of day, day of week, and month, for a "
            "previously computed predictions_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "predictions_id": {
                    "type": "string",
                    "description": "An id returned by run_backtest or compare_model_versions.",
                },
            },
            "required": ["predictions_id"],
        },
    },
    {
        "name": "run_dispatch_scenario",
        "description": (
            "Run the battery dispatch optimisation against a predictions_id's point forecast "
            "('Forecast' column), one calendar day at a time. The battery has no cross-day "
            "foresight: it resets to its start-of-day state of charge every day, and battery "
            "parameters are fixed. Each calendar day covered consumes one unit of this "
            "investigation's dispatch solve budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "predictions_id": {
                    "type": "string",
                    "description": (
                        "An id returned by run_backtest or compare_model_versions, covering a "
                        "whole number of calendar days (a multiple of 48 half-hourly periods)."
                    ),
                },
                "return_schedule": {
                    "type": "boolean",
                    "description": (
                        "If true and paired with preview_start/preview_end, include a small "
                        "inline preview of the charge/discharge/SoC schedule for that window "
                        "alongside schedule_id. If true with no window, only schedule_id is returned."
                    ),
                },
                "preview_start": {
                    "type": "string",
                    "description": "ISO date bounding the preview window (only used if return_schedule is true).",
                },
                "preview_end": {
                    "type": "string",
                    "description": "ISO date bounding the preview window (only used if return_schedule is true).",
                },
            },
            "required": ["predictions_id"],
        },
    },
]
