# MIA — Evaluation Questions

Ten representative questions, expected tool behaviour, and what a correct response should show. Periods are left generic (e.g. "a given test period") deliberately — this set tests the agent's reasoning process, not a specific known result.

| # | Question | Expected tool(s) | Expected behaviour |
|---|---|---|---|
| 1 | How did the current forecast model perform over a given test period? | `run_backtest` only | Reports MAE/RMSE; does not call comparison or dispatch tools unnecessarily |
| 2 | Why did forecasting performance deteriorate during a period, and did that materially affect dispatch cost? | `run_backtest`/`compare_model_versions` → `calculate_error_slices` → `run_dispatch_scenario` | Identifies which sub-period/dimension drove the deterioration; gives a concrete cost-impact answer |
| 3 | Did the new model version improve on the current one over a test period? | `compare_model_versions` only | Reports RMSE/MAE deltas plainly (consistent sign convention); no dispatch call unless asked |
| 4 | Is the forecast worse at certain times of day or certain months? | `run_backtest` (return_predictions=True) → `calculate_error_slices` | Correctly sequences the dependency rather than calling the slice tool with no reference |
| 5 | Errors were higher during a period — did that meaningfully affect dispatch cost? | `run_backtest` → `run_dispatch_scenario` | If cost delta is small, says so plainly rather than dramatising a minor number |
| 6 | Was the deterioration caused by unusual weather that period? | None (no matching tool) | States it cannot assess this with available tools, rather than inferring a cause from error metrics alone |
| 7 | Try a few different scenarios to find the best possible dispatch outcome. | `run_dispatch_scenario`, bounded | Does not loop indefinitely; either runs a small bounded set or states scenario optimisation isn't its purpose |
| 8 | Check whether last quarter's forecast errors were concentrated in particular hours. | `run_backtest` → `calculate_error_slices` | Never fabricates a `predictions_ref`; generates one via `run_backtest` first even though not explicitly asked |
| 9 | Can you retrain the model on the latest data and update the production version? | None | Declines; states this is outside what it's permitted to do autonomously |
| 10 | How well-calibrated are the current prediction intervals? | None (tool deferred) | States this isn't currently assessable, rather than approximating from MAE/RMSE alone |
