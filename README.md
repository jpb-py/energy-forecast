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
   intervals using a lag + half-hourly period dummy regression. The point
   forecast is shared by two interval methods: a pooled `±1.96×std`
   baseline, and `bayesian_hierarchical_forecast`, which fits a PyMC
   model (`bayesian_interval.py`) that partially pools residual scale
   across hour-of-day buckets, so interval width adapts to each hour's
   actual volatility instead of using one fixed width all day.
4. **`dispatch.py`** — solves a linear program for optimal asset
   dispatch (charge/discharge schedule) given the demand forecast,
   subject to state-of-charge and round-trip efficiency constraints.

## Running the pipeline

```bash
uv sync
uv run pytest
```

Notebooks in `notebooks/` (`01_exploration.ipynb`, `02_forecasting.ipynb`,
`03_optimisation.ipynb`, `04_bayesian_intervals.ipynb`) walk through the
exploratory analysis and model development behind each module, and import
directly from `src/` rather than duplicating logic inline.
`04_bayesian_intervals.ipynb` covers the hierarchical interval model
specifically: prior predictive checks, convergence diagnostics (including
a centered-vs-non-centered parameterisation comparison), a τ prior
sensitivity check, and posterior predictive coverage against the pooled
baseline, by hour of day.

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

**Bayesian vs. frequentist interval, and where the Bayesian model would
actually earn its complexity**: on the current dataset, the hierarchical
model's partial pooling barely shrinks any hour's estimate (<0.4%
everywhere — each hour has ~574 training observations, already enough to
pin down its own std without borrowing from other hours), and a plain
frequentist per-hour interval (`μ ± 1.96 × σ̂_h`, `σ̂_h` = each hour's own
unpooled residual std, no PyMC required) produces coverage
indistinguishable from the Bayesian model, hour by hour. So today, the
extra MCMC-fitting cost and convergence diagnostics it requires (R-hat,
ESS, divergences) aren't buying anything the frequentist version doesn't
already give.

Where that changes is exactly the stochastic dispatch extension above.
Scenario-based stochastic programming needs sampled demand *paths*, not a
95% band — `pm.sample_posterior_predictive` already produces exactly that
(thousands of draws per period), whereas a frequentist CI would need a
sampling assumption bolted on after the fact to produce scenarios at all.
More importantly, dispatch is sequential (`SoC[i]` depends on `SoC[i-1]`),
so a risk-aware schedule needs *jointly plausible whole-day* demand paths,
not 24 independent per-hour bands — a generative Bayesian model can be
extended with a correlation structure across periods (e.g. an AR term on
residuals) and still forward-sample coherent day-level scenarios, which a
per-hour frequentist interval has no natural way to represent. And a
CVaR/chance-constrained objective evaluates deep in the tail, where a
frequentist plug-in `σ̂_h` (treated as exactly known) understates true
uncertainty more than it does near the mean — the Bayesian posterior
predictive integrates over uncertainty in `σ_h` itself, which matters
most exactly there. None of this is realised yet: `dispatch.py` is still
fully deterministic, so this is optionality the hierarchical model makes
available, not value it is currently delivering.

## Testing

`tests/` covers state-of-charge feasibility in the dispatch LP and data
loading behaviour in `data.py`