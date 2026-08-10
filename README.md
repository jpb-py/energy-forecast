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

## Agent experiment: Model Investigation Assistant

As an extension to the core pipeline, this project includes a small autonomous
agent — the Model Investigation Assistant (MIA) — that answers open-ended
diagnostic questions about forecast and dispatch performance (e.g. *"why did
forecasting performance deteriorate during a period, and did that materially
affect dispatch cost?"*) by deciding for itself which read-only diagnostic
tool to call, and in what order, rather than following a fixed script.

**Scope, deliberately kept narrow**: MIA evaluates and compares forecast
models and dispatch outcomes that already exist. It cannot create, train, or
modify a model, and no tool can write to the repo or a saved artifact — a
design decision made up front, not a limitation discovered later.

**Tools**: `run_backtest`, `compare_model_versions`, `calculate_error_slices`,
and `run_dispatch_scenario`, each specified with explicit inputs, outputs,
and read-only guarantees, plus a hard cap on tool calls and on the
underlying optimisation solver's invocation count per query.

**Evaluation**: a ten-question set covering both straightforward and
deliberately ambiguous/out-of-scope requests, run manually against the built
agent. This surfaced several genuine failure modes — a model narrating an
intended action instead of taking it and stopping; silently substituting an
easier question instead of flagging an ambiguous one; a token-limit cutoff
producing a silent blank response instead of a visible error — each traced
to a specific cause and fixed.

The most significant finding wasn't about the agent's reasoning at all: an
early dispatch-comparison result looked plausible and internally consistent
(a less accurate forecast appeared to produce a better dispatch outcome),
with a coherent causal story attached. It turned out to be wrong for two
compounding reasons — an ambiguously-named return field (`total_cost`, which
was actually profit, later renamed to `total_profit`/`realized_profit`) that
led the agent to read even its own correct-at-the-time numbers backwards,
and a genuine scoring bug where cost/profit was computed against the same
forecast used to build the schedule rather than against actual demand.
Independently re-deriving the result after fixing both reversed the
conclusion. It's a concrete example of why agent outputs need verification,
not just plausibility review, regardless of how sound the agent's own
tool-selection and reasoning process looks — and a reminder that ambiguity
can hide in something as small as a field name, not just in underlying logic.

Full evaluation writeup: `MIA_Failure_Modes.md`, `MIA_Lessons_Learned.md`.

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