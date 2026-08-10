import itertools
from dataclasses import dataclass, field

import pandas as pd

from energy_forecast.forecast import forecast_with_interval, naive_seasonal_forecast

FORECAST_FUNCTIONS = {
    "linear_lagged": forecast_with_interval,
    "seasonal_naive": naive_seasonal_forecast,
}


def resolve_forecast_fn(forecast_fn_id: str):
    try:
        return FORECAST_FUNCTIONS[forecast_fn_id]
    except KeyError:
        known = ", ".join(sorted(FORECAST_FUNCTIONS))
        raise KeyError(f"Unknown forecast_fn_id '{forecast_fn_id}'. Known ids: {known}.") from None


class ResultStore:
    """Session-scoped, in-memory store mapping opaque ids to real pandas objects
    (predictions/schedule DataFrames), so tool JSON can reference them by id
    instead of inlining large tables."""

    def __init__(self):
        self._objects = {}
        self._counter = itertools.count(1)

    def put(self, prefix: str, obj) -> str:
        object_id = f"{prefix}_{next(self._counter):03d}"
        self._objects[object_id] = obj
        return object_id

    def get(self, object_id: str):
        try:
            return self._objects[object_id]
        except KeyError:
            raise KeyError(
                f"Unknown id '{object_id}'. It may belong to a different session, or was never created."
            ) from None


@dataclass
class Session:
    """Per-investigation state: the raw data, the results store, and the two
    hard guardrail counters (total tool calls, underlying dispatch solves)."""

    raw_data: pd.DataFrame
    store: ResultStore = field(default_factory=ResultStore)
    total_calls_used: int = 0
    total_calls_budget: int = 10
    dispatch_solves_used: int = 0
    dispatch_solves_budget: int = 50
