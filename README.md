# Flexible Asset Dispatch under Demand Uncertainty

Given day-ahead UK electricity demand forecasts with uncertainty bounds,
this project determines the optimal dispatch schedule for a flexible
asset (e.g. battery storage), balancing forecast demand against price
and asset constraints.

## Pipeline

1. **`data.py`** — loads and cleans NESO half-hourly demand data,
   building a proper datetime index from settlement date and period.
2. **`features.py`** — engineers time-based features (hour, day of
   week, month, weekend flag) and lag features (1, 48, 336 periods)
   used by the forecast model.
3. **`forecast.py`** — produces a demand forecast with 95% prediction
   intervals using a lag + half-hourly period dummy regression.
4. **`dispatch.py`** — solves a linear program for optimal asset
   dispatch (charge/discharge schedule) given the demand forecast,
   subject to state-of-charge and round-trip efficiency constraints.

## Running the pipeline

```bash
uv sync
uv run pytest
```

Notebooks in `notebooks/` (`01_exploration.ipynb`, `02_forecasting.ipynb`,
`03_optimisation.ipynb`) walk through the exploratory analysis and model
development behind each module, and now import directly from `src/`
rather than duplicating logic inline.

## Possible extensions

The current dispatch model treats the demand forecast as a single point
estimate with prediction intervals used for reporting, not as an input
to the optimisation itself. A natural extension would be a **stochastic
or risk-aware dispatch formulation** — e.g. scenario-based stochastic
programming over sampled demand paths, or a chance-constrained /
CVaR-based objective — so the dispatch decision explicitly accounts for
forecast uncertainty rather than optimising against the mean forecast
alone.

## Testing

`tests/` covers state-of-charge feasibility in the dispatch LP and data
loading behaviour in `data.py`